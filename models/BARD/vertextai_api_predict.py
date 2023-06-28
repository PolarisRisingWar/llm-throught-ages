#Linux环境配置方式，参考https://cloud.google.com/sdk/docs/install?hl=zh-cn#linux：
#第一步：pip install google-cloud-aiplatform
#第二步：wget -P llm-throught-ages/models/BARD https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-436.0.0-linux-x86_64.tar.gz
#（↑具体下哪儿无所谓）
#第三步：tar -xf llm-throught-ages/models/BARD/google-cloud-cli-436.0.0-linux-x86_64.tar.gz
#第四步：./google-cloud-sdk/install.sh（是否支持随便选，第二个填Y，第三个随便选）
#第五步：打开新终端，输入./google-cloud-sdk/bin/gcloud init
#第步：在这里看项目ID，替换下方project_id变量：

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