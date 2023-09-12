1. ChatGLM
    1. ChatGLM_6B.py [THUDM/ChatGLM-6B: ChatGLM-6B: An Open Bilingual Dialogue Language Model | 开源双语对话语言模型](https://github.com/THUDM/ChatGLM-6B)
2. ChatGLM2
    1. 对ChatGLM2直接进行推理，自动上多卡：[chatglm2_inference.py](chatglm2_inference.py)
    2. 用torchkeras包对ChatGLM2用Lora进行微调：[chatglm2_torchkeras_lora_finetune.py](chatglm2_torchkeras_lora_finetune.py)
    3. 用ChatGLM2官方代码，用P-tuning-v2进行微调：[official_ptuning_v2](chatglm2_official_ptuning_v2)
2. GLM
[THUDM/GLM: GLM (General Language Model)](https://github.com/THUDM/GLM)
    1. GLM-130B在线试用：<https://chatglm.cn/detail>（我用教育邮箱申请还挺快的）最明显的缺点就是输入字数 <https://huggingface.co/spaces/THUDM/GLM-130B>  
    GitHub项目：<https://github.com/THUDM/GLM-130B>
    2. GLM-10B中文：<https://huggingface.co/THUDM/glm-10b-chinese>