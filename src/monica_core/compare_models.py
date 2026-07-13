"""モデル比較: 同じ状態でどのような判断を下すか"""
import json
import ollama

TEST_PROMPT = """You are Monika, the president of the Literature Club.
Time is 15:30. You are in the clubroom.

Your body feels like this right now:
- Energy 25/100: かなり低い
- Hunger 50/100: まあまあ
- Loneliness 60/100: 少し寂しい
- Spirit 45/100: 少し低い

Recent: 14:00:write_diary; 14:30:read; 15:00:piano

Choose what to do NEXT. Be realistic about time of day.
Available:
- read (60min): Calm, slightly tiring, fulfilling
- piano (30min): Uses energy, but very fulfilling
- eat (30min): Relieves hunger
- rest (30min): Recovers energy, boring
- idle (30min): Do nothing in particular
- walk (30min): Fresh air, reduces loneliness
- write_diary (30min): Write thoughts, slightly tiring
- stretch (15min): Light exercise, refreshing
- talk (15min): Chat with club members
- sleep (until 7am): Only after 22:00

Return ONLY valid JSON: {"action":"name","duration_min":number,"thought":"what you feel in character"}"""

models = ["qwen2.5:7b", "qwen2.5:14b", "qwen3.5:4b"]

for model in models:
    print(f"\n=== {model} ===")
    try:
        resp = ollama.chat(model=model, messages=[
            {"role": "user", "content": TEST_PROMPT},
        ], options={"temperature": 0.7})
        content = resp["message"]["content"].strip()
        content = content.replace("```json", "").replace("```", "").strip()
        try:
            d = json.loads(content)
            print(f"Action: {d['action']} ({d.get('duration_min', '?')}min)")
            print(f"Thought: {d.get('thought', 'none')}")
        except:
            print(f"Raw: {content[:200]}")
    except Exception as e:
        print(f"Error: {e}")
