from key import *

import clueai

cl = clueai.Client(ClueAI_KEY,check_api_key=True)
prompt= '''用户：上班时候老是犯困怎么办？
小元：'''
prediction = cl.generate(
            model_name='ChatYuan-large',
            prompt=prompt)
                   
print('prediction: {}'.format(prediction.generations[0].text))