改自官方代码：<https://github.com/THUDM/ChatGLM2-6B/blob/main/ptuning>

需要：`pip install rouge_chinese nltk jieba datasets`

先运行train.sh进行微调
1. train.json和valid.json每行是一个序列化的JSON对象，其中content键是原文，summary键是生成文本

（测试部分等我试运行完再补）

其他参考资料：
1. [ChatGLM2-6B 在 ModelWhale 平台的部署与微调教程 - Heywhale.com](https://www.heywhale.com/mw/project/64984a7b72ebe240516ae79c)
2. [ChatGLM-6B微调的一点心得 - 知乎](https://zhuanlan.zhihu.com/p/646497519)：这篇是用ChatGLM的，但是应该跟ChatGLM2差不多吧