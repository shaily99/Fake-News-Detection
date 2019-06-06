from watson_developer_cloud import ToneAnalyzerV3

from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import \
    Features, RelationsOptions, CategoriesOptions, ConceptsOptions, KeywordsOptions, SentimentOptions, EntitiesOptions


tone_analyzer = ToneAnalyzerV3(
    version='2017-09-21',
    iam_apikey='Mn7hB_yZm2FVFWWhovYknkYpC1JG3VEFAXVgnQOgCt8D',
    url='https://gateway-tok.watsonplatform.net/tone-analyzer/api'
)


natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2018-11-16',
    iam_apikey='QNE9N5fCRCa_xb2y4fyBgOwPdSpZ8U2pkqHuY8t5YFQC',
    url='https://gateway-tok.watsonplatform.net/natural-language-understanding/api'
)

def getres(file_loc):
    with open(file_loc, 'r+') as f:
        head = f.readline()
        content = f.read()
        req = '<html><body><h2>{0}</h2>{1}</body></html>'.format(
                head,
                content
        )
        text = head + content
    tone_res = tone_analyzer.tone(
    req,
    content_type='text/html'
    ).get_result()

    res = natural_language_understanding.analyze(
    html = req,
    features=Features(categories=CategoriesOptions(limit=1),
                      concepts=ConceptsOptions(limit=5),
                      keywords=KeywordsOptions(limit=5, sentiment=True, emotion=True),
                      sentiment=SentimentOptions(),
                      # entities=EntitiesOptions(limit=5, mentions=True, sentiment=True, emotion=True),
                      ),
    ).get_result()
    sentiment = res["sentiment"]["document"]["score"]
    concepts = [(concepts["text"], concepts["relevance"]) for concepts in res["concepts"]]
    categories  = (res["categories"][0]["label"].split("/"), res["categories"][0]["score"])
    keywords = [(keywords["text"],keywords["relevance"]) for keywords in res["keywords"]]
    tones = [(tone["tone_id"],tone["score"]) for tone in tone_res["document_tone"]["tones"]]
    return (sentiment, concepts, keywords, tones, text)


