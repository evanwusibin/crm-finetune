"""
修复 Qwen3 thinking 模式：在 system prompt 末尾加 /no_think
"""

import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

for fname in ["crm_train.jsonl", "crm_test.jsonl"]:
    fpath = os.path.join(DATA_DIR, fname)
    lines = []
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            msgs = data["messages"]
            if msgs[0]["role"] == "system":
                old_content = msgs[0]["content"]
                if "/no_think" not in old_content:
                    msgs[0]["content"] = old_content + " /no_think"
            lines.append(json.dumps(data, ensure_ascii=False))
    
    with open(fpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    
    print(f"✅ {fname}: {len(lines)} 条，已添加 /no_think")

print("\n验证第一条：")
with open(os.path.join(DATA_DIR, "crm_train.jsonl"), "r", encoding="utf-8") as f:
    data = json.loads(f.readline().strip())
    print(f"  system: {data['messages'][0]['content']}")
