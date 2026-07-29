"""ワークステーション監視GUI (192.168.1.58)
温度/電力/GPU/CPU/メモリ/ログを表示"""
import tkinter as tk
from tkinter import ttk
import threading
import paramiko
import json
import re

HOST = "192.168.1.58"
USER = "koushi"
PASS = "koushi0928"
REFRESH_MS = 5000

class WSMonitor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("WS Monitor - 1080Ti Workstation")
        self.root.geometry("1100x750")
        self.root.configure(bg="#1a1a2e")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#1a1a2e")
        style.configure("TLabel", background="#1a1a2e", foreground="#e0e0e0", font=("Consolas", 10))
        style.configure("TLabelframe", background="#1a1a2e", foreground="#e0e0e0")

        self._build_ui()
        self._update()
        self.root.mainloop()

    def _build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # === GPU section ===
        gpu_frame = ttk.LabelFrame(main, text="GPU (x2 GTX 1080 Ti)", padding=10)
        gpu_frame.pack(fill=tk.X, pady=(0, 5))
        self.gpu_labels = []
        for i in range(2):
            f = ttk.Frame(gpu_frame)
            f.pack(fill=tk.X, pady=2)
            lbl = ttk.Label(f, text=f"GPU{i}: --")
            lbl.pack(side=tk.LEFT)
            self.gpu_labels.append(lbl)
        # Total power
        self.power_label = ttk.Label(gpu_frame, text="Total Power: --", font=("Consolas", 11, "bold"))
        self.power_label.pack(anchor=tk.W, pady=(4, 0))

        # === System resources ===
        sys_frame = ttk.LabelFrame(main, text="System", padding=10)
        sys_frame.pack(fill=tk.X, pady=5)
        self.sys_labels = {}
        for k in ["CPU", "Memory", "Swap"]:
            f = ttk.Frame(sys_frame)
            f.pack(fill=tk.X)
            lbl = ttk.Label(f, text=f"{k}: --")
            lbl.pack(side=tk.LEFT)
            self.sys_labels[k] = lbl

        # === Ollama status ===
        ollama_frame = ttk.LabelFrame(main, text="Ollama", padding=10)
        ollama_frame.pack(fill=tk.X, pady=5)
        self.ollama_label = ttk.Label(ollama_frame, text="--")
        self.ollama_label.pack()

        # === Connection status ===
        self.status_label = ttk.Label(main, text="", foreground="#888")
        self.status_label.pack(anchor=tk.W, pady=(2, 0))

        # === Log viewer ===
        log_frame = ttk.LabelFrame(main, text="journalctl (last 20)", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text = tk.Text(log_frame, height=12, bg="#0d0d1a", fg="#c0c0c0",
                                 font=("Consolas", 8), state=tk.DISABLED, wrap=tk.WORD)
        scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _ssh_cmd(self, cmd: str) -> str:
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(HOST, username=USER, password=PASS, timeout=8)
            _, stdout, stderr = ssh.exec_command(cmd, timeout=10)
            out = stdout.read().decode(errors="replace")
            err = stderr.read().decode(errors="replace")
            ssh.close()
            return out + err
        except Exception as e:
            return f""

    def _update(self):
        t = threading.Thread(target=self._fetch_and_update, daemon=True)
        t.start()
        self.root.after(REFRESH_MS, self._update)

    def _fetch_and_update(self):
        try:
            ok = True

            # GPU info
            gpu_out = self._ssh_cmd(
                "nvidia-smi --query-gpu=index,name,temperature.gpu,fan.speed,power.draw,utilization.gpu,memory.used,memory.total"
                " --format=csv,noheader,nounits"
            )
            if gpu_out:
                total_power = 0.0
                lines = gpu_out.strip().split("\n")
                for i, line in enumerate(lines):
                    if i < 2 and line:
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 8:
                            idx, name, temp, fan, power, util, mem_u, mem_t = parts[:8]
                            try:
                                total_power += float(power)
                            except ValueError:
                                pass
                            txt = f"GPU{idx}: {name} | {temp}C | fan {fan}% | {power}W | util {util}% | mem {mem_u}/{mem_t} MiB"
                            self.gpu_labels[i].configure(text=txt)
                self.power_label.configure(text=f"Total Power: {total_power:.1f} W")

            # CPU / Memory
            sys_out = self._ssh_cmd(
                "top -bn1 2>/dev/null | awk '/%Cpu/ {u=$2; s=$4; printf \"CPU: %.1f%%\", u+s}'"
                " && echo"
                " && free -h | awk '/^Mem/ {printf \"Memory: %s/%s\", $3, $2}'"
                " && echo"
                " && free -h | awk '/^Swap/ {printf \"Swap: %s/%s\", $3, $2}'"
            )
            if sys_out:
                for line in sys_out.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("CPU"):
                        self.sys_labels["CPU"].configure(text=line)
                    elif line.startswith("Memory"):
                        self.sys_labels["Memory"].configure(text=line)
                    elif line.startswith("Swap"):
                        self.sys_labels["Swap"].configure(text=line)

            # Ollama
            ps_out = self._ssh_cmd("ollama ps 2>&1 | tail -n +2 | head -5")
            self.ollama_label.configure(text=ps_out.strip() or "No model loaded")

            # Logs
            log_out = self._ssh_cmd(
                "journalctl -u ollama --no-pager -n 30 --output=short-iso 2>&1 | tail -20"
            )
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.delete("1.0", tk.END)
            self.log_text.insert(tk.END, log_out[:4000] or "(no logs)")
            self.log_text.configure(state=tk.DISABLED)

            self.status_label.configure(text="OK", foreground="#4caf50")

        except Exception as e:
            self.status_label.configure(text=f"Error: {e}", foreground="#f44336")

if __name__ == "__main__":
    WSMonitor()
