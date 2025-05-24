from datasets import load_dataset
import json

# 載入資料集
dataset = load_dataset("ruiruiw/HuatuoGPT_datasets", split="train")

# 轉換函數：處理每筆資料
def convert_conversations(example):
    conv = example["conversations"]
    prompt = ""
    response = ""

    for i in range(0, len(conv), 2):
        if (
            i + 1 < len(conv)
            and conv[i]["from"] in ["human", "patient"]
            and conv[i + 1]["from"] in ["gpt", "doctor"]
        ):
            prompt += f"[User] {conv[i]['value']}\n[Assistant] {conv[i + 1]['value']}\n"
            response = conv[i + 1]["value"]

    return {
        "prompt": prompt.strip(),
        "output": response.strip()
    }

# ⚠️ 使用 remove_columns 移除原始欄位（如 conversations）
formatted_dataset = dataset.map(
    convert_conversations,
    remove_columns=dataset.column_names  # 把原本的欄位都移除，只保留回傳的 prompt/output
)

# 儲存為 JSONL 格式
with open("huatuo_prompt_output.jsonl", "w", encoding="utf-8") as f:
    for row in formatted_dataset:
        json.dump(row, f, ensure_ascii=False)
        f.write("\n")

print("✅ 資料轉換完成，共處理筆數：", len(formatted_dataset))
