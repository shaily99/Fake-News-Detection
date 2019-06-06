from pprint import pprint
import os
import json
from watson_developer_cloud import DiscoveryV1

discovery = DiscoveryV1(
    version="2018-12-03",
    iam_apikey='6QYKBEdxyh4mJvh6yRyzg5mP8O2ke1mA9sw_S73sjXTj',
    url="https://gateway-lon.watsonplatform.net/discovery/api"
)

ENV = '1917047f-502f-4e06-9140-9c272daf86af'
CONF = '3c289112-b899-43a2-9f79-c630853c7c88'
COLLECT = '608b85a7-ba90-45d8-93cb-becbc99b4588'

DocIDlist = ['01574b96-8436-4a68-aeee-fcba6eac40bd']


def get_col(col):
    collection = discovery.get_collection(ENV, col).get_result()
    print(json.dumps(collection, indent=2))


def post_file(file_, path_):
    global DocIDlist
    with open(os.path.join(path_, file_)) as fileinfo:
        add_doc = discovery.add_document(
                    ENV,
                    COLLECT,
                    file=fileinfo,
                    file_content_type='text/html'
                    ).get_result()
    pprint(add_doc)
    DocIDlist.append(str(add_doc["document_id"]))


def get_file(doc):
    doc_info = discovery.get_document_status(ENV, COLLECT, doc).get_result()
    print(json.dumps(doc_info, indent=2))


def del_file(doc):
    delete_doc = discovery.delete_document(ENV, COLLECT, doc).get_result()
    print(json.dumps(delete_doc, indent=2))


quer_calls = 0


def queryCol(**kwargs):
    global quer_calls
    quer = discovery.query(
            ENV,
            COLLECT,
            **kwargs
    )
    qdict = quer._to_dict()
    quer_calls += 1
    pprint(qdict)
    with open('quer{0}.txt'.format(quer_calls), 'w+') as f:
        f.write(repr(qdict))



""" Script Goes Here """

# post_file('real3999.html', '.')
# post_file('real4000.html', '.')


# get_col(COLLECT)
# print('\n\nSTARTING ADDITION\n\n')
# dataset_path = '../teamName/training/celebrityDataset/fake'
# for fyl in os.listdir(dataset_path):
#     post_file(fyl, dataset_path)



# pprint(DocIDlist)
# Writes Document Ids
# with open('doc_ids.json', 'w+') as f:
#     f.write(json.dumps(DocIDlist))

""" Open interactively and query from fake news """

tone_res = './tone_res/celebrityDataset/fake'
# for file_ in os.listdir(dataset_path):
#     with open(os.path.join(tone_res, file_), 'r+') as f:
#         info = json.dump(f)
