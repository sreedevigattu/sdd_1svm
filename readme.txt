learn.py 
Trains a OneSVM based classifier for the specified category and creates <category>class.pkl files 
python learn.py E:/Sree/netalytics/SensitiveDataDiscovery/data/code/python/ unstructured text tfidf python 1000 show


categorizer.py
Categorizes the files in the specified folder and writes results into elastic search

categorize.py
Categorizes the given file/content using the classifiers in the class folder 

do.py
filemonitor watches the configured folders. If there are file additions/deletions etc., these are 
added to a processing queue. The files are then processed/categorized by multiple worker threads.


# Imports
keras, sklearn, matplotlib, nltk, tika, pandas, numpy, joblib, pathlib, pprint, 
sdd.text, sdd.utility, sdd.categorize_nn, sdd.process
keras.layers, keras.models, keras.utils, keras.preprocessing.sequence,  keras.preprocessing.text
sklearn.metrics
matplotlib.pyplot
nltk, nltk.corpus, nltk.stem, en_core_web_md
tika, tika.detector, tika.parser
pandas,numpy, joblib
pathlib, pprint, random, re, datetime, string,os, sys

To find cpu usage on linux
ps aux | grep -i sdd
pidstat -h -r -u -v -p 9095 20 | grep 9095
