from pprint import pprint
import os
import json
import csv



name_ = 'keyw_celeb_fake.csv'
f = open(name_, 'w+')
outWrite = csv.writer(f)


path_ = './natlang_responses/celebrityDataset/fake'
for file_ in os.listdir(path_):
    with open(os.path.join(path_, file_), 'r+') as f:
        res = json.load(f)
    row10 = [file_, res['categories'][0]['label'], res['categories'][0]['score']]
    for concept in res['concepts']:
        row10.append(concept['text'])
        row10.append(concept['relevance'])

    if len(row10) != 12:
        print('LACKS CONCEPT')
    else:
        print(row10)
    outWrite.writerow(row10)


f.close()
