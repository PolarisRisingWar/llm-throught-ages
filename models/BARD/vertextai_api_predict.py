#Linux环境配置方式，参考https://cloud.google.com/sdk/docs/install?hl=zh-cn#linux：
#第一步：pip install google-cloud-aiplatform
#第二步：wget -P llm-throught-ages/models/BARD https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-436.0.0-linux-x86_64.tar.gz
#（↑具体下哪儿无所谓）
#第三步：tar -xf llm-throught-ages/models/BARD/google-cloud-cli-436.0.0-linux-x86_64.tar.gz
#第四步：./google-cloud-sdk/install.sh（是否支持随便选，第二个填Y，第三个随便选）
#第五步：打开新终端，输入./google-cloud-sdk/bin/gcloud init
#（如果出现代理问题：可以直接根据输出提示设置代理，也可以参考https://cloud.google.com/sdk/docs/proxy-settings?hl=zh-cn实现设置：
#gcloud config set proxy/type http
#gcloud config set proxy/address [PROXY_IP_ADDRESS]
#gcloud config set proxy/port 7890
#第六步：gcloud auth application-default login根据提示登录自己的谷歌账号并提供token（参考https://cloud.google.com/docs/authentication/provide-credentials-adc?hl=zh-cn#local-dev）
#第步：在这里看项目ID，替换下方project_id变量：https://console.cloud.google.com/home/dashboard

#import os

#os.environ['HTTP_PROXY'] = 'http://:7890'
#os.environ['HTTPS_PROXY'] = 'https://:7890'


import vertexai
from vertexai.language_models import TextGenerationModel

from key import *

vertexai.init(project=project_id, location="us-central1")
parameters = {
    "temperature": 0.2,
    "max_output_tokens": 256,
    "top_p": 0.8,
    "top_k": 40
}
model = TextGenerationModel.from_pretrained("text-bison")
response = model.predict(
    """Vertex AI, this platform is so hard to use. Are there any other tutorials easier to understand?""",
    **parameters
)
print(f"Response from Model: {response.text}")