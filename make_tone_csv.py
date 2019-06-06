from pprint import pprint
import os
import json
import csv



name_ = 'emotion_fakeNews_fake.csv'
f = open(name_, 'w+')
outWrite = csv.writer(f)


path_ = './tone_responses/fakeNewsDataset/fake'
for file_ in os.listdir(path_):
    with open(os.path.join(path_, file_), 'r+') as f:
        res = json.load(f)
    row10 = []
    for tone in res['document_tone']['tones']:
        row10.append(tone['tone_id'])
        row10.append(tone['score'])
        if len(row10) > 10:
            break
    while len(row10)<10:
        row10.append('NaN')
    print(row10)
    outWrite.writerow(row10)


f.close()
