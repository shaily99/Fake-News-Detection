from pprint import pprint
import os
import json
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import \
    Features, RelationsOptions, CategoriesOptions, ConceptsOptions, KeywordsOptions, SentimentOptions, EntitiesOptions

natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2018-11-16',
    iam_apikey='QNE9N5fCRCa_xb2y4fyBgOwPdSpZ8U2pkqHuY8t5YFQC',
    url='https://gateway-tok.watsonplatform.net/natural-language-understanding/api'
)


# Set Variables and make sure folder exists in responses also!
DATASET = 'fakeNewsDataset/fake'


path_ = os.path.join('../teamName/training/', DATASET)
for file_ in os.listdir(path_):
    with open(os.path.join(path_, file_), 'r+') as f:
        req = '<html><body><h2>{0}</h2>{1}</body></html>'.format(
                f.readline().strip(),
                ''.join([line.strip() for line in f.readlines()])
        )
    response = natural_language_understanding.analyze(
        html = req,
        features=Features(categories=CategoriesOptions(limit=1),
                          concepts=ConceptsOptions(limit=5),
                          keywords=KeywordsOptions(limit=5, sentiment=True, emotion=True),
                          sentiment=SentimentOptions(),
                          entities=EntitiesOptions(limit=5, mentions=True, sentiment=True, emotion=True),
                          ),
    ).get_result()
    new_file_name = ''.join(file_.split('.')[:-1]) + '.json'
    with open(os.path.join('./responses/', DATASET, new_file_name), 'w+') as outF :
        outF.write(json.dumps(response, indent=2))

    pprint(response)
