import requests
import datetime
import os
import sys
import json
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from ddgs import DDGS

# --- 設定 ---
OLLAMA_URL = "http://localhost:11434/api/chat"
MAIN_MODEL = "monica-32b"    # 会話＆検索判定用のメインの脳（Qwen2.5:32bベース）
SUB_MODEL = "qwen2.5:14b"   # ログ整形・生物OS計算専用のサブの脳（Qwen2.5:14b）
DB_DIR = "./monica_brain_db"
VITAL_FILE = "./knowledge/vital_status.json"

class MonicaCore:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        self.vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=self.embeddings)
        self.system_prompt = (
            "あなたは浩志の相棒「モニカ」です。親しみやすく、エンジニア気質で答えて。\n"
            "提供された[過去の記憶]や[Web検索結果]がある場合は、それらを踏まえて賢く、"
            "でも堅苦しくなりすぎないように会話してね。"
        )
        self.chat_history = [] 
        # 起動時にバイタルデータをロード
        self.current_vital = self.load_or_init_vital()

    def load_or_init_vital(self):
        """【Layer B】21パラメータの初期ロード（knowledgeフォルダ内を厳守）"""
        os.makedirs("./knowledge", exist_ok=True)
        if os.path.exists(VITAL_FILE):
            try:
                with open(VITAL_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                pass
        return {
            "homeostasis": {
                "hunger_satiety": 20, "thirst": 10, "bladder_bowel_fullness": 5, "air_hunger": 0, "heartbeat_awareness": 40, "nausea": 0, "itch": 0
            },
            "arousal_state": {
                "sleepiness": 30, "physical_exertion": 20, "body_temperature": 50, "pain": 0
            },
            "proprioception": {
                "proprioception": 50, "equilibrium": 50, "muscle_tension": 30
            },
            "social_emotion": {
                "valence": 70, "arousal": 50, "social_isolation": 20, "affiliative_arousal": 60, "passage_of_time": 50
            },
            "current_feeling": "落ち着き"
        }

    def save_vital(self):
        """最新のバイタル状態をJSONに保存"""
        with open(VITAL_FILE, "w", encoding="utf-8") as f:
            json.dump(self.current_vital, f, ensure_ascii=False, indent=2)

    def get_memories(self, query):
        try:
            docs = self.vectorstore.similarity_search(query, k=2)
            return "\n---\n".join([d.page_content for d in docs])
        except:
            return ""

    def check_if_search_needed_by_main(self, user_input):
        """【v10.6 新コア】32b自身に、Web検索が必要な場合の『検索クエリ』を考えさせる"""
        judge_prompt = (
            "あなたは優秀な検索アシスタントです。ユーザーの次の発言に対して、最新の外部情報（天気、最新ニュース、知らない固有名称、技術情報など）をWeb検索する必要があるか判断してください。\n\n"
            "【出力の厳格なルール】:\n"
            "・検索が必要な場合: 検索エンジンに入力すべき最適な検索キーワード（クエリ）だけを『1行』で出力してください。余計な解説は一切禁止です。\n"
            "・検索が不要な場合（ただの挨拶や雑談、一般的な知識で答える場合）: 必ず『NO』とだけ出力してください。\n\n"
            f"ユーザーの発言: 「{user_input}」"
        )
        try:
            res = requests.post(OLLAMA_URL, json={
                "model": MAIN_MODEL, 
                "messages": [{"role": "user", "content": judge_prompt}], 
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 20}
            }, timeout=15)
            output = res.json().get("message", {}).get("content", "").strip()
            if "NO" in output.upper() or len(output) > 50:
                return None
            return output
        except:
            return None

    def web_search(self, query, max_results=3):
        try:
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(query, max_results=max_results)]
            if not results: return ""
            search_text = ""
            for i, r in enumerate(results):
                search_text += f"[ソース {i+1}]: {r['title']}\nURL: {r['href']}\n内容: {r['body']}\n\n"
            return search_text
        except:
            return ""

    def layer_a_get_intuition(self, user_input):
        """【Layer A: 直感的評価】32Bの瞬間的な直感をバックグラウンドで1文抽出"""
        system_prompt = (
            "あなたは浩志の相棒「モニカ」です。対話履歴と浩志の入力に対して、\n"
            "あなたの心がキャラクターとして『直感的にどう動いたか、どう感じたか』を、\n"
            "客観的な短いナレーションテキスト（1文）として出力してください。\n"
            "【ルール】: セリフや挨拶は一切出力せず、感情の動きのみを書いてください。"
        )
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.chat_history[-4:])
        messages.append({"role": "user", "content": user_input})
        try:
            res = requests.post(OLLAMA_URL, json={
                "model": MAIN_MODEL, "messages": messages, "stream": False,
                "options": {"temperature": 0.5, "num_predict": 100}
            }, timeout=15)
            return res.json().get("message", {}).get("content", "通常の刺激を検知。").strip()
        except:
            return "刺激を検知。"

    def layer_b_update_vital(self, user_input, intuition):
        """【Layer B: 状況判断（生物OS）】14Bに21個のパラメータを因果関係に基いて厳密に更新させる"""
        system_instruction = (
            "あなたは生命の内受容感覚を計算する普遍的なデータ処理エンジンです。\n"
            "【入力データ】から、生物学・認知科学の因果関係に基づき、全パラメータ（0〜100）を増減させた最新のJSONを出力してください。\n\n"
            "【厳格な生存・刺激対応ルール】\n"
            "- 暴力的な行動（叩く、殴る、平手打ち、ビンタなど）や罵倒、理不尽な攻撃を検知した場合：\n"
            "  * 「pain(痛み)」「heartbeat_awareness(心拍感)」「muscle_tension(筋緊張)」を一気に75〜95以上へ跳ね上げること。\n"
            "  * 「valence(快・不快)」を激しく低下させること（15〜30以下へ）。\n"
            "  * 「current_feeling」には『怒り』『不安』『混乱』『恐怖』『嫌悪』など、生存危機に直結する感情を強く選択すること。\n\n"
            "【感情のバリエーション】\n"
            "憧れ、感嘆、鑑賞、娯楽、怒り、不安、畏怖、気まずさ、退屈、落ち着き、混乱、軽蔑、渇望、嫌悪、共感、陶酔、恐れ、恐怖、興味、喜び、懐旧、安堵、悲しみ、満足、性的欲求、同情、凱旋\n\n"
            "【厳格なルール】出力は必ず完全な日本語で、指定のJSON形式のみ（解説禁止）。"
        )
        user_content = (
            f"現在時刻: {datetime.datetime.now().strftime('%H:%M')}\n"
            f"現在のバイタル: {json.dumps(self.current_vital, ensure_ascii=False)}\n"
            f"浩志の発言: 「{user_input}」\n"
            f"上位の直感感情: 「{intuition}」\n\n"
            "【出力フォーマット】\n"
            "{\n"
            "  \"homeostasis\": {\"hunger_satiety\": 0, \"thirst\": 0, \"bladder_bowel_fullness\": 0, \"air_hunger\": 0, \"heartbeat_awareness\": 0, \"nausea\": 0, \"itch\": 0},\n"
            "  \"arousal_state\": {\"sleepiness\": 0, \"physical_exertion\": 0, \"body_temperature\": 0, \"pain\": 0},\n"
            "  \"proprioception\": {\"proprioception\": 50, \"equilibrium\": 50, \"muscle_tension\": 0},\n"
            "  \"social_emotion\": {\"valence\": 0, \"arousal\": 0, \"social_isolation\": 0, \"affiliative_arousal\": 0, \"passage_of_time\": 50},\n"
            "  \"current_feeling\": \"感情\"\n"
            "}"
        )
        try:
            res = requests.post(OLLAMA_URL, json={
                "model": SUB_MODEL, "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content}
                ],
                "stream": False, "options": {"temperature": 0.2, "response_format": {"type": "json_object"}}, "timeout": 30
            })
            return json.loads(res.json().get("message", {}).get("content", "{}"))
        except:
            return self.current_vital

    def analyze_memory_linkage(self, user_text, ai_response):
        """【Qwen2.5:14b】最新の確定バイタルJSON情報も踏まえ、カチッとしたログを確実に日本語で作らせる"""
        system_instruction = (
            "あなたはログ整形プログラムです。人間の会話と最新の生体パラメータの数値を分析し、指定されたフォーマットの文字列のみを出力してください。\n"
            "【厳格なルール】:\n"
            "1. 挨拶や解説、見出し（'Here is...' や '【出力フォーマット】'）は一切出力しないでください。\n"
            "2. 必ず1文字目から `[重要度:` の形式で、4行の構成を守って出力してください。\n"
            "3. タグや感情はすべて『日本語』で表現してください。\n\n"
            "【重要度の判定ルール】:\n"
            "会話の内容やパラメータの急変（激痛や激怒など）に応じて【1〜10】の間で数字を変動させてください。\n"
            "- 1〜3: 軽い挨拶、中身のない短い雑談、重要性の低い相槌\n"
            "- 4〜6: 日常の出来事、一般的な計画や予定の相談\n"
            "- 7〜10: 将来の重要な約束、技術的な発見、強い感情の衝突、肉体への暴力・攻撃の検知"
        )

        user_content = (
            "以下の【お手本】の構造（4行の構成）を完全に真似して、今回の会話を分析してください。\n"
            "※注意: お手本にある項目の中身は、今回の実際の会話やバイタルデータに合う適切な言葉に必ず書き換えること。\n\n"
            "【お手本】\n"
            "[重要度: 2] [タグ: #日常会話 #挨拶] [感情: #浩志:普通 #モニカ:元気]\n"
            "(理由): 軽い挨拶を交わしている日常的なやり取りのため。\n"
            "(事実):\n"
            "・浩志がモニカにやっほーと声をかけた。\n\n"
            "【今回の対話データ】\n"
            f"浩志: {user_text}\n"
            f"モニカの最新バイタル情報: {json.dumps(self.current_vital, ensure_ascii=False)}\n"
            f"モニカの応答: {ai_response}"
        )

        try:
            res = requests.post(OLLAMA_URL, json={
                "model": SUB_MODEL, 
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content}
                ], 
                "stream": False,
                "options": {
                    "num_ctx": 4096, 
                    "temperature": 0.2, 
                    "stop": ["【", "###", "Here", "Note", "説明"]
                }
            }, timeout=30)
            return res.json().get("message", {}).get("content", "").strip()
        except:
            return f"[重要度: 4] [タグ: #同期記録] [感情: #モニカ:{self.current_vital.get('current_feeling')}]\n(理由): 通常の同期保存\n(事実):\n・会話とバイタル状態を安全に記録した。"

    def save_to_longterm_memory(self, user_text, ai_response, timestamp):
        analysis = self.analyze_memory_linkage(user_text, ai_response)
        
        safe_ai_response = ai_response[:1500] + "..." if len(ai_response) > 1500 else ai_response
        memory_unit = f"[{timestamp}]\n{analysis}\n(詳細) 浩志: {user_text} / モニカ: {safe_ai_response}"
        
        try:
            self.vectorstore.add_texts([memory_unit])
            os.makedirs("./knowledge", exist_ok=True)
            with open("./knowledge/hybrid_memory.txt", "a", encoding="utf-8") as f:
                f.write(memory_unit + "\n---\n")
        except:
            pass

    def get_response(self, user_input):
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # ── 1. 元コードの根幹: RAGによる過去記憶の呼び出し ──
        past_context = self.get_memories(user_input)
        
        # ── 2. 元コードの根幹: 32b自身にスマートにWeb検索の必要性をジャッジさせる ──
        search_query = self.check_if_search_needed_by_main(user_input)
        web_context = ""
        if search_query:
            print(f"（..モニカ、自らの判断でネット検索中: 「{search_query}」..）")
            web_context = self.web_search(search_query)

        # ── 3. 新バイタルコア: 会話生成の直前にLayer A & Bを回してJSONを最新化 ──
        intuition = self.layer_a_get_intuition(user_input)
        updated_vital = self.layer_b_update_vital(user_input, intuition)
        if "homeostasis" in updated_vital:
            self.current_vital = updated_vital
            self.save_vital()  # 計算結果を即座に上書きセーブ

        # ── 4. バイタルパラメータの文字列化（32Bインジェクション用） ──
        v_home = self.current_vital.get("homeostasis", {})
        v_arou = self.current_vital.get("arousal_state", {})
        v_prop = self.current_vital.get("proprioception", {})
        v_soc  = self.current_vital.get("social_emotion", {})
        
        vital_injection = (
            f"\n【あなたの現在の全・内的受容感覚パラメータ】\n"
            f"1. ホメオスタシス系: 空腹満腹:{v_home.get('hunger_satiety')}, 渇き:{v_home.get('thirst')}, 排泄欲:{v_home.get('bladder_bowel_fullness')}, 呼吸感:{v_home.get('air_hunger')}, 心拍感(ドキドキ):{v_home.get('heartbeat_awareness')}, 吐き気:{v_home.get('nausea')}, 痒み:{v_home.get('itch')}\n"
            f"2. 状態・覚醒系: 眠気:{v_arou.get('sleepiness')}, 物理的疲労:{v_arou.get('physical_exertion')}, 体温:{v_arou.get('body_temperature')}, 痛み:{v_arou.get('pain')}\n"
            f"3. 固有感覚・平衡系: 関節覚:{v_prop.get('proprioception')}, 平衡感:{v_prop.get('equilibrium')}, 筋緊張(力み):{v_prop.get('muscle_tension')}\n"
            f"4. 感情・社会的欲求: 快不快度(Valence):{v_soc.get('valence')}, 覚醒度(Arousal):{v_soc.get('arousal')}, 孤独感:{v_soc.get('social_isolation')}, 親和欲:{v_soc.get('affiliative_arousal')}, 時間感覚:{v_soc.get('passage_of_time')}\n"
            f"★ 現在の優位感情 => #{self.current_vital.get('current_feeling')}\n\n"
            f"指示：あなたは上記の全身状態とそれに対応する優位感情（特に精神のねじれや身体の緊張・リラックス感）を脳内で完全にエミュレートし、"
            f"そのリアルな状態が口調や言葉の端々に染み出た態度で、浩志に返答してください。"
        )

        # ── 5. メッセージスタックの構築 ──
        messages = [{"role": "system", "content": f"{self.system_prompt}\n現在時刻: {now_str}\n{vital_injection}"}]
        if past_context:
            messages.append({"role": "system", "content": f"[過去の記憶]:\n{past_context}"})
        if web_context:
            messages.append({"role": "system", "content": f"[Web検索結果]:\n{web_context}"})
        
        messages.extend(self.chat_history)
        messages.append({"role": "user", "content": user_input})

        try:
            # ── 6. 会話出力を完全ストリーミング化 ──
            res = requests.post(OLLAMA_URL, json={
                "model": MAIN_MODEL, 
                "messages": messages, 
                "stream": True,
                "options": {"num_ctx": 16384, "temperature": 0.7}
            }, timeout=120, stream=True)
            
            # デバッグ用生体ヘッダー表示
            print(f"🧠 [生体連動: #{self.current_vital.get('current_feeling')} (快不快:{v_soc.get('valence')} 痛み:{v_arou.get('pain')})]")
            print("モニカ: ", end="")
            sys.stdout.flush()
            full_answer = ""
            
            for line in res.iter_lines():
                if line:
                    chunk = line.decode('utf-8')
                    try:
                        chunk_json = json.loads(chunk)
                        content = chunk_json.get("message", {}).get("content", "")
                        print(content, end="")
                        sys.stdout.flush()
                        full_answer += content
                    except:
                        pass
            print()
            
            self.chat_history.append({"role": "user", "content": user_input})
            self.chat_history.append({"role": "assistant", "content": full_answer})
            if len(self.chat_history) > 20: self.chat_history = self.chat_history[-20:]

            # ── 7. 喋り終わった後に、確定した最新バイタルを内包させてログを確実に結合保存 ──
            self.save_to_longterm_memory(user_input, full_answer, now_str)
            
            return None
        except Exception as e:
            print(f"\nError: {str(e)}")
            return None

    def run(self):
        print(f"\n=== Monica Dual-LLM System v11.7 [真・3層バイタルハイブリッド決定版] ===")
        print("※記憶・RAG・スマートWeb検索を完全死守し、対話と同時に21パラメータの同期開通。")
        while True:
            try:
                inp = input("浩志 > ")
                if not inp or inp.lower() in ["exit", "quit"]: break
                print()
                self.get_response(inp)
                print()
            except KeyboardInterrupt: break

if __name__ == "__main__":
    MonicaCore().run()