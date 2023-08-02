ChatGPT官网：<https://chat.openai.com/>

ChatGPT提供了API可以直接调用。  
1. [GPT-3.5-example1.py](GPT-3.5-example1.py)：调用ChatGPT-3.5的API，输入一个问题，输出一个回答。为了适应生产环境的需要进行了以下改进：①增加了重试策略 ②自动检测输入token数是否超过4096，从而自动决定调用哪个上下文版本