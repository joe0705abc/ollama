import sys
import ollama

sys.stdout.reconfigure(encoding='utf-8')  # 讓 print() 支援 Unicode

response = ollama.chat(model='deepseek-r1:1.5b', messages=[{'role': 'user', 'content': '你好，Ollama！'}])

print(response['message']['content'])  # 正常輸出中文[Running] 