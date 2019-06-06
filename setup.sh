#!/bin/bash

apt-get install python3-pip
pip3 install wheel
pip3 install setuptools
pip3 install watson_developer_cloud
pip3 install numpy
pip3 install nltk pandas sklearn
python3 -c 'import nltk; nltk.download("wordnet"); nltk.download("stopwords");'

