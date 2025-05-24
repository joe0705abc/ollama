from datasets import load_dataset
import json

# 載入資料集
dataset = load_dataset("ruiruiw/HuatuoGPT_datasets", split="train")

# 自訂轉換函數：抓最後一輪 human + gpt 對話
def convert_conversations(example):
    conv = example["conversations"]
    prompt = ""
    response = ""

    # 擷取所有 human/gpt 對話並格式化
    for i in range(0, len(conv), 2):
        if i + 1 < len(conv) and conv[i]["from"] == "human" and conv[i+1]["from"] == "gpt":
            prompt += f"[User] {conv[i]['value']}\n[Assistant] {conv[i+1]['value']}\n"
            response = conv[i+1]["value"]  # 抓最後一次 gpt 回應當 output

    return {
        "prompt": prompt.strip(),
        "output": response.strip()
    }

# 應用轉換
formatted_dataset = dataset.map(convert_conversations)

# 儲存為 JSONL（適合 LoRA 使用）
with open("huatuo_prompt_output.jsonl", "w", encoding="utf-8") as f:
    for row in formatted_dataset:
        json.dump(row, f, ensure_ascii=False)
        f.write("\n")

print("✅ 資料轉換完成，共處理筆數：", len(formatted_dataset))
