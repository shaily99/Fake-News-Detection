# Fake-News-Detection

The following files are included in the directory provided:

* classifier.sh: You are allowed to write your code in the language of your choice; use this script as a wrapper such that it can be executed from the command line as follows:

    ./classifier <path to article file>

    The article file will follow the same format as the articles in the training set (i.e., first line is the title and the rest is the content). This script is required to provide an exit status of 0 if the article is real, and 1 if it is fake. Presently, the exit status is random. As of now, it executes a dummy script (dummy.py) that outputs prime numbers upto the input argument on the command line, and exits with a random exit status (0 or 1)
    Eg: $ ./classifier 20
    2 3 5 7 11 13 17 19
    $

* evaluate: This script will be used to evaluate the time and space complexity of your code.
    Eg: $ ./evaluate ./classifier 10000
    Mem  : 7556
    Time : 8.24
    $

    Note: this script suppresses output from your classifier, and reports "Command exited with non-zero status 1" if your classifier reports the article as real, and only outputs the peak memory and total time elapsed between the launch and exit of your classifier.

* setup.sh: This file is expected to perform all the environmental setup required to run your code on the provided virtual machine, including installation of all required libraries, and downloading of any special packages (eg: nltk.download("wordnet")). It will be run as:

    $ sudo -H ./setup.sh

    Immediately after this, we will be running your classifier (i.e., ./classifier) and evaluating its performance using the "evaluate" script.

    The VM is a .vdi (VirtualBox Disk Image) file with Ubuntu 18.04.1 LTS x64. To run it, you need to have installed VirtualBox 6.0 from https://www.virtualbox.org/wiki/Downloads (download the appropriate package for your system). Simply double-click the pravega-ibm-hackathon.vbox file; ignore the error if it pops up. Launch VirtualBox, and you should see an entry for this VM. Double-click it to run it. You may need to set a different display scaling under View -> Virtual Screen 1 -> Scale to x% (autoscaled output).

    Username: strange
    Password: password

    It has been set up with some basic packages, including python, nltk, pandas and sklearn. The installation script used was:

---
#!/bin/bash

    apt update
    apt install -y build-essential gcc
    apt install -y python3-pip python-pip
    pip install wheel
    pip install setuptools
    pip install nltk pandas sklearn
    pip3 install wheel
    pip3 install setuptools
    pip3 install nltk pandas sklearn
    python -c 'import nltk;nltk.download("wordnet"); nltk.download("stopwords")'
    python3 -c 'import nltk;nltk.download("wordnet"); nltk.download("stopwords")'
    apt install -y jupyter-notebook firefox
---


---------------------------

Link to dataset and API responses : 
https://drive.google.com/open?id=1NfHIjJBN-HjrbatK4EBHVr1DjFO-Otl6

Dataset citation:

@article{Perez-Rosas18Automatic,
author = {Ver\’{o}nica P\'{e}rez-Rosas, Bennett Kleinberg, Alexandra Lefevre, Rada Mihalcea},
title = {Automatic Detection of Fake News},
journal = {International Conference on Computational Linguistics (COLING)},
year = {2018}
}

