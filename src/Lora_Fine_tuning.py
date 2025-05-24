from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import torch

# === 設定參數 ===
MODEL_NAME = "THUDM/chatglm2-6b"
DATA_PATH = "./huatuo_prompt_output.jsonl"
OUTPUT_DIR = "./chatglm2-huatuo-lora"

# === 載入 tokenizer 和模型 ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    device_map="auto",
    load_in_8bit=True  # 省記憶體
)

# === 加入 LoRA 配置 ===
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["query_key_value"]  # ChatGLM2-specific
)

model = get_peft_model(model, lora_config)

# === 載入資料集（使用 JSONL 格式） ===
dataset = load_dataset("json", data_files={"train": DATA_PATH})["train"]

# === Tokenizer 處理函數 ===
def preprocess(example):
    full_prompt = example["prompt"]
    response = example["output"]
    input_text = full_prompt

    tokenized = tokenizer(
        input_text,
        max_length=1024,
        padding="max_length",
        truncation=True
    )
    labels = tokenizer(
        response,
        max_length=1024,
        padding="max_length",
        truncation=True
    )["input_ids"]

    tokenized["labels"] = labels
    return tokenized

tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# === 設定訓練參數 ===
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    output_dir=OUTPUT_DIR,
    save_total_limit=2,
    save_steps=100,
    remove_unused_columns=False
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# === 執行訓練 ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ LoRA 微調完成！")
