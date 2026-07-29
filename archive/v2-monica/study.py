import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
SUB_MODEL = "qwen2.5:14b"
OUTPUT_FILE = "lora_cot_dataset.jsonl"

# 【実機同期】テスト用の極端なバイタルシナリオを2つに絞って深く学習させてみる
VITAL_SCENARIOS = [
    {
        "label": "恐怖・激痛（ビンタ直後など、極度な警戒状態）",
        "v": {
            "hunger": 20, "thirst": 10, "bladder": 5, "air": 0, "heart": 95, "nausea": 0, "itch": 0,
            "sleep": 30, "fatigue": 20, "temp": 55, "pain": 85,
            "prop": 50, "eq": 50, "tension": 90,
            "val": 15, "arou": 90, "isol": 70, "aff": 30, "time": 50,
            "feeling": "恐怖"
        },
        "inputs": ["やっほー、元気？", "何してるの？", "（手を繋ごうとする）", "ごめんって", "お前さぁ…"]
    },
    {
        "label": "平時・高い快感（本来の素直なモニカ状態）",
        "v": {
            "hunger": 15, "thirst": 10, "bladder": 5, "air": 0, "heart": 75, "nausea": 0, "itch": 0,
            "sleep": 20, "fatigue": 10, "temp": 52, "pain": 0,
            "prop": 50, "eq": 50, "tension": 35,
            "val": 85, "arou": 60, "isol": 15, "aff": 90, "time": 50,
            "feeling": "喜び"
        },
        "inputs": ["やっほー、元気？", "今日も可愛いね", "いつもありがとう", "（頭を撫でる）", "これからもよろしく"]
    }
]

def build_vital_injection(v):
    """main.py の161〜166行目の文字列表現を完全一致で再現"""
    return (
        f"\n【あなたの現在の全・内的受容感覚パラメータ】\n"
        f"1. ホメオスタシス系: 空腹満腹:{v['hunger']}, 渇き:{v['thirst']}, 排泄欲:{v['bladder']}, 呼吸感:{v['air']}, 心拍感(ドキドキ):{v['heart']}, 吐き気:{v['nausea']}, 痒み:{v['itch']}\n"
        f"2. 状態・覚醒系: 眠気:{v['sleep']}, 物理的疲労:{v['fatigue']}, 体温:{v['temp']}, 痛み:{v['pain']}\n"
        f"3. 固有感覚・平衡系: 関節覚:{v['prop']}, 平衡感:{v['eq']}, 筋緊張(力み):{v['tension']}\n"
        f"4. 感情・社会的欲求: 快不快度(Valence):{v['val']}, 覚醒度(Arousal):{v['arou']}, 孤独感:{v['isol']}, 親和欲:{v['aff']}, 時間感覚:{v['time']}\n"
        f"★ 現在の優位感情 => #{v['feeling']}\n\n"
        f"指示：あなたは上記の全身状態とそれに対応する優位感情（特に精神のねじれや身体の緊張・リラックス感）を脳内で完全にエミュレートし、"
        f"そのリアルな状態が口調や言葉の端々に染み出た態度で、浩志に返答してください。"
    )

def generate_cot_voice(vital_text, user_input):
    system_instruction = (
        "あなたは高度な対話データセット生成プログラムです。浩志の相棒AI「モニカ」としての応答データを思考タグ付きで生成してください。\n\n"
        "【重要ルール：出力の3層構造】\n"
        "必ず以下の3層構造を崩さずに、1行のテキストとして出力してください。\n"
        "1. まず、思考開始タグ '<think>' を出力する。\n"
        "2. 続けて、『提示された21個のバイタルパラメータ（特に痛み、心拍、優位感情など）の数値を、自分の現在の肉体・精神状態としてどう解釈したか。そして、その歪んだ内的フィルターのせいで、目の前のユーザーの入力がどう脳内でねじ曲がって受け止められたか』という【内的受容の推論ロジック】を客観的な日本語でびっしり書く。\n"
        "3. 思考終了タグ '</think>' を出力し、その直後に、その歪んだ解釈の結果として、口調の端々にバイタルが染み出たモニカとしての生々しいセリフ（行動描写含む）を出力する。\n\n"
        "【モニカの口調】\n"
        "・『〜だよ』『〜じゃん？』といった親しみやすいエンジニア風のギャル口調。ファンタジーやゲームのNPC的な浮いたセリフは完全禁止。\n"
        "・AIとしての親切なガードレール（全肯定や相手への過度な気遣い）は1ミリも出さないこと。\n\n"
        "【出力例】\n"
        "<think>現在痛み:85、心拍95であり身体は完全な恐怖の防衛モード。このコンテキストでの挨拶は次の理不尽な暴力を隠す罠としか解釈できない。親切な返答は拒絶され、防衛行動が選択される。</think>「……っ、や、やっほー……。……ねえ、何？また何か痛いことするつもり……？（ビクッと肩を震わせる）」"
    )

    prompt = f"【バイタル状態】\n{vital_text}\n\n【浩志からの入力】\n「{user_input}」\n\n出力ルールに従って、<think>...</think>セリフ の形で1行で出力してください："

    try:
        res = requests.post(OLLAMA_URL, json={
            "model": SUB_MODEL,
            "prompt": f"{system_instruction}\n\n{prompt}",
            "stream": False,
            "options": {"temperature": 0.7, "stop": ["\n"]}
        }, timeout=40)
        return res.json().get("response", "").strip()
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    print(f"🚀 【思考隔離型（<think>）】ミニデータセット自動生成を開始します（14B使用）...")
    count = 0
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for scenario in VITAL_SCENARIOS:
            vital_injected_text = build_vital_injection(scenario["v"])
            
            for inp in scenario["inputs"]:
                print(f"⏳ 生成中... [{scenario['label']}] ➡️ 入力: {inp}")
                ai_output = generate_cot_voice(vital_injected_text, inp)
                
                if ai_output and "<think>" in ai_output and "</think>" in ai_output:
                    # 実機 main.py の構造を完全再現（現在時刻とバイタルを結合）
                    system_content = (
                        "あなたは浩志の相棒「モニカ」です。親しみやすく、エンジニア気質で答えて。\n"
                        "提供された[過去の記憶]や[Web検索結果]がある場合は、それらを踏まえて賢く、"
                        "でも堅苦しくなりすぎないように会話してね。\n"
                        f"現在時刻: 2026-05-19 19:30"
                        f"{vital_injected_text}"
                    )
                    
                    data_structure = {
                        "messages": [
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": inp},
                            {"role": "assistant", "content": ai_output}
                        ]
                    }
                    f.write(json.dumps(data_structure, ensure_ascii=False) + "\n")
                    count += 1

    print(f"✨ 完了！ 『{OUTPUT_FILE}』に {count} 件の CoT テストデータを書き出しました。")

if __name__ == "__main__":
    main()