import os
from pprint import pprint
import json
from watson_developer_cloud import ToneAnalyzerV3


tone_analyzer = ToneAnalyzerV3(
    version='2017-09-21',
    iam_apikey='Mn7hB_yZm2FVFWWhovYknkYpC1JG3VEFAXVgnQOgCt8D',
    url='https://gateway-tok.watsonplatform.net/tone-analyzer/api'
)


# Set Variable
DATASET = 'fakeNewsDataset/fake'

path_ = os.path.join('../teamName/training/', DATASET)
for file_ in os.listdir(path_):
    new_file_name = ''.join(file_.split('.')[:-1]) + '.json'
    if new_file_name in os.listdir(os.path.join('./tone_res/', DATASET)):
        print('Skipped Existing File!')
        continue
    with open(os.path.join(path_, file_), 'r+') as f:
        req = '<html><body><h2>{0}</h2>{1}</body></html>'.format(
            f.readline().strip(),
            ''.join([line.strip() for line in f.readlines()])
        )
        # pprint(req)
    res = tone_analyzer.tone(
        req,
        content_type='text/html'
    ).get_result()
    with open(os.path.join('./tone_res/', DATASET, new_file_name), 'w+') as outF :
        outF.write(json.dumps(res, indent=2))

    pprint(res)
