#!/bin/bash
# Monica VitalOS - setup script for fresh environment

python3 -m venv .venv
.venv/bin/pip install -e .

if [ ! -f .env ]; then
    echo "OPENCODE_ZEN_API_KEY=sk-..." > .env
    echo "MONIKA_MODEL=big-pickle" >> .env
    echo ""
    echo ".env を作成しました。APIキーを設定してください:"
    echo "  nano .env"
fi

echo "完了！起動: .venv/bin/python3 -m monica_core.simulate_llm"
