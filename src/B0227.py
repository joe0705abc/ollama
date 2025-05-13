import sys
import ollama

sys.stdout.reconfigure(encoding='utf-8')  # 讓 print() 支援 Unicode


# 選擇模型
model = "deepseek-r1:1.5b"

print("🤖 Ollama Chat - 輸入 'exit' 來離開")

chat_history = []  # 存儲對話歷史

while True:
    user_input = input("👤 你: ")
    
    if user_input.lower() == "exit":
        print("👋 再見！")
        break

    # 新增用戶輸入到歷史記錄
    chat_history.append({"role": "user", "content": user_input})
    
    # 獲取 Ollama 回應
    response = ollama.chat(model=model, messages=chat_history)
    
    # 解析回應
    bot_reply = response['message']['content']
    print(f"🤖 Ollama: {bot_reply}\n")

    # 新增 Ollama 回應到歷史記錄
    chat_history.append({"role": "assistant", "content": bot_reply})
    
    print(chat_history)