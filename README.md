本项目致力于总结各种LLM的简单最佳实践代码。

理论部分大概会主要写在博客上：[LLM的实践古往今来（持续更新ing…）](https://blog.csdn.net/PolarisRisingWar/article/details/130533565)  
GitHub项目负责实现，包括直接推理和微调。  
目前看来应该不会包括预训练。

本项目不包括非自研的应用服务。这个可以参考我写的博文：[在线LLM应用集锦（持续更新ing...）](https://blog.csdn.net/PolarisRisingWar/article/details/131115434)

以下是我已经用过或正在用的模型。按照首字母顺序排列：

- [BARD / Vertex AI / PaLM](models/BARD/)
- [BERT](models/BERT/)
- [ChatGPT / GPT-3 / GPT-3.5 / Whisper AI / DALL-E](models/GPT_family/ChatGPT_API)
- [ERNIE](models/ERNIE/)
- [GLM](models/GLM)：调用过程中出现了bug，正在找原作者提问：[用transformers包，下载文件到本地后无法加载AutoTokenizer · Issue #181 · THUDM/GLM](https://github.com/THUDM/GLM/issues/181)
- [LLaMA](models/LLaMA/)：填了表，正在泡waiting list
- [通义千问](models/tongyi/)
- [讯飞星火](models/Spark/)
- [源1.0](models/yuan1.0/)

按照任务排列：
- 文本分类
[BERT](models/BERT/TC/)
- 序列标注
    - 命名实体识别
- 文本生成
    - 抽取式摘要
    - 生成式摘要：可以先参考我开的另一个项目 [PolarisRisingWar/text_summarization_chinese: 各大文本摘要模型-中文文本可运行的解决方案](https://github.com/PolarisRisingWar/text_summarization_chinese)（已经完全耦合了，目前不太确定这两边要怎么解耦，这个以后再考虑吧先把东西写出来再说）
    - 推理
    - 文本补全&问答
        - 仅支持直接推理：
            - 仅支持网页客户端的：[BARD](models/BARD/)，[文心一言](models/ERNIE)
            - 支持网页客户端和API调用的：[讯飞星火](models/Spark/)
            - 支持API调用的：[源1.0](models/yuan1.0/)
        - 支持直接推理和云端微调：[ChatGPT / GPT-3 / GPT-3.5](models/GPT_family/ChatGPT_API)
        - 支持本地部署：ChatGLM
- 音视频转文字
[Whisper AI](models/GPT_family/ChatGPT_API)
[通义听悟](models/tongyi)

本文在撰写过程中使用的集成接口包括但不限于：
- transformers
- textgen [shibing624/textgen: TextGen: Implementation of Text Generation models, include LLaMA, BLOOM, GPT2, BART, T5, SongNet and so on. 文本生成模型，实现了包括LLaMA，ChatGLM，BLOOM，GPT2，Seq2Seq，BART，T5，UDA等模型的训练和预测，开箱即用。](https://github.com/shibing624/textgen)