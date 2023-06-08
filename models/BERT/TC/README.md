可参考我写的博文：[用huggingface.transformers.AutoModelForSequenceClassification在文本分类任务上微调预训练模型_诸神缄默不语的博客-CSDN博客](https://blog.csdn.net/PolarisRisingWar/article/details/123939061)

- [在通用BERT checkpoint的基础上，使用Trainer微调，使用datasets包官方数据集](bert_tc_trainer.py)
- [在通用BERT checkpoint的基础上，使用原生PyTorch微调，使用datasets包官方数据集，使用transformers.get_scheduler，用datasets包内置的评估指标](bert_tc_native_pytorch.py)
- [在通用BERT checkpoint的基础上，使用原生PyTorch微调，使用自定义数据集，用sklearn包的评估指标](bert_tc_native_pytorch2.py)（可参考我写的博文：[完全小白如何运行人生中的第一个Bert文本分类代码_诸神缄默不语的博客-CSDN博客](https://blog.csdn.net/PolarisRisingWar/article/details/129644300)）

通用BERT checkpoint，我用过能直接用的：
- <https://huggingface.co/bert-base-uncased>
- <https://huggingface.co/bert-base-cased>
- <https://huggingface.co/bert-base-chinese>