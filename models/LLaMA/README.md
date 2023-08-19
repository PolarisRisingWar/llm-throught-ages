LLaMA huggingface.transformers官网：<https://huggingface.co/docs/transformers/main/model_doc/llama>  
权重理论上需要填表申请（我还在泡池子）  

1. LLaMA_community1.py：直接使用社区公开的LLaMA权重
2. LLaMA_community2.py：也是直接使用社区公开的LLaMA权重，但经中文优化：[ymcui/Chinese-LLaMA-Alpaca: 中文LLaMA&Alpaca大语言模型+本地CPU/GPU训练部署 (Chinese LLaMA & Alpaca LLMs)](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
3. pyllama1.py：使用[juncongmoo/pyllama: LLaMA: Open and Efficient Foundation Language Models](https://github.com/juncongmoo/pyllama)包
[[代码学习]也尝试一下LLaMa-7B - 知乎](https://zhuanlan.zhihu.com/p/622927692)
4. [textgen_llama1.py](textgen_llama1.py)：使用textgen包实现测试（英文版，即原版）
5. [textgen_llama2.py](textgen_llama2.py)：使用textgen包实现测试（中文版，使用LLaMA的checkpoint和chinese-alpaca-lora-7b）（很慢）
6. [textgen_llama3.py](textgen_llama3.py)：使用textgen包实现测试（中文版，使用LLaMA的checkpoint和chinese-alpaca-lora-7b手动合并后的checkpoint）（明显比上一条快很多）
（以上3个测试文件的参考代码都是<https://github.com/shibing624/textgen/blob/main/examples/gpt/predict_demo.py>，从单样本输出换成多样本输出都是直接在`model.predict`后面的列表里加样本就可以）
7. [textgen_llama4.py](textgen_llama4.py)：使用textgen包实现下游任务微调并生成测试结果（运行命令是python （英文版）
8. LLaMA2待更新。中文版已出：<https://huggingface.co/LinkSoul/Chinese-Llama-2-7b> <https://github.com/ymcui/Chinese-LLaMA-Alpaca-2> <https://huggingface.co/ziqingyang/chinese-llama-2-7b> <https://github.com/ymcui/Chinese-LLaMA-Alpaca-2> <https://github.com/CVI-SZU/Linly> <https://huggingface.co/wenge-research/yayi-13b-llama2> <https://github.com/yangjianxin1/Firefly> <https://huggingface.co/OpenBuddy/openbuddy-llama2-13b-v8.1-fp16>

其他参考资料：
1. [测试了下llama的效果（附带权重、怎么跑） - 知乎](https://zhuanlan.zhihu.com/p/613419608)