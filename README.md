本项目致力于总结各种LLM的简单最佳实践代码。

理论部分大概会主要写在博客上：[LLM的实践古往今来（持续更新ing…）](https://blog.csdn.net/PolarisRisingWar/article/details/130533565)  
GitHub项目负责实现，包括直接推理和微调。  
目前看来应该不会包括预训练。

以下是我已经用过或正在用的模型。按照首字母顺序排列：

- [BARD](models/BARD/)
- [BERT](models/BERT/)
- [ChatGPT / GPT-3 / GPT-3.5](models/GPT_family/ChatGPT_API)
- [ERNIE](models/ERNIE/)
- [GLM](models/GLM)：调用过程中出现了bug，正在找原作者提问：[用transformers包，下载文件到本地后无法加载AutoTokenizer · Issue #181 · THUDM/GLM](https://github.com/THUDM/GLM/issues/181)
- [LLaMA](models/LLaMA/)：填了表，正在泡waiting list
- [源1.0](models/yuan1.0/)

按照任务排列：
- 文本分类
[BERT](models/BERT/TC/)
- 序列标注
    - 命名实体识别
- 文本生成
    - 抽取式摘要
    - 生成式摘要：可以先参考我开的另一个项目 [PolarisRisingWar/text_summarization_chinese: 各大文本摘要模型-中文文本可运行的解决方案](https://github.com/PolarisRisingWar/text_summarization_chinese)（目前不太确定这两边要怎么解耦，这个以后再考虑吧先把东西写出来再说）
    - 推理
    - 文本补全