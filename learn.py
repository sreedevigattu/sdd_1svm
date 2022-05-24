import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys, os, random
import process, text

''' 
python learn.py E:/Sree/netalytics/SensitiveDataDiscovery/data/fusionalgo/FIN/ structured value tfidf finance 1000 show > out/fin_1svm_tfidf_1000.txt
python learn.py E:/Sree/netalytics/SensitiveDataDiscovery/data/fusionalgo/FIN/ unstructured value tfidf finance 1000 show > out/fin_1svm_unsvalue_tfidf_1000.txt
python learn.py E:/Sree/netalytics/SensitiveDataDiscovery/data/fusionalgo/FIN/ unstructured text tfidf finance 1000 show > out/fin_1svm_unstext_tfidf_1000.txt

python learn.py E:/Sree/netalytics/SensitiveDataDiscovery/data/fusionalgo/BBPS/ structured value tfidf bank 1000 show > out/bank_1svm_tfidf_1000.txt

python learn.py E:/Sree/netalytics/SensitiveDataDiscovery/data/fusionalgo/BBPS/ unstructured text tfidf bank 1000 show > out/bank_bbps_1svm_unstext_tfidf_1000.txt

python learn.py E:/Sree/netalytics/SensitiveDataDiscovery/data/fusionalgo/PVR_APRIL/ unstructured text tfidf pvr 1000 show > out/pvr_1svm_ustext_tfidf_1000.txt
python learn.py E:/Sree/netalytics/SensitiveDataDiscovery/data/fusionalgo/PVR_APRIL/ unstructured value tfidf pvr 1000 show > out/pvr_1svm_usvalue_tfidf_1000.txt

python learn.py E:/Sree/netalytics/SensitiveDataDiscovery/data/fusionalgo/ unstructured text tfidf bank 1000 show > out/bank_1svm_unstext_tfidf_1000.txt
python learn.py E:/Sree/netalytics/SensitiveDataDiscovery/data/fusionalgo/ unstructured value tfidf bank 1000 show 

python learn.py E:/Sree/netalytics/SensitiveDataDiscovery/data/pwd/ unstructured text tfidf pwd 1000 show > out/pwd_1svm_unstext_tfidf_1000.txt
python learn.py E:/Sree/netalytics/SensitiveDataDiscovery/data/pwd/ unstructured text count pwd 1000 show > out/pwd_1svm_unstext_count_1000.txt
python learn.py E:/Sree/netalytics/SensitiveDataDiscovery/data/pwd/ unstructured text word2vec pwd 1000 show > out/pwd_1svm_unstext_w2v_1000.txt

python learn.py E:/Sree/netalytics/SensitiveDataDiscovery/data/code/java/ unstructured text tfidf java 1000 show > out/java_1svm_unstext_tfidf_1000.txt
python learn.py E:/Sree/netalytics/SensitiveDataDiscovery/data/code/python/ unstructured text tfidf python 1000 show > out/python_1svm_unstext_tfidf_1000.txt
'''

def visualize(featcounts, df):  
    #print(plt.style.available)
    mpl.style.use(['ggplot'])

    # Which are the most frequently occurring words?
    from wordcloud import WordCloud, STOPWORDS
    wordcloud = WordCloud().generate_from_frequencies(featcounts)
    #fig1 = plt.figure(class_+"wc")
    plt.title(class_ + "- Frequent Words")
    plt.box(on=False)
    plt.grid(b=False)
    plt.imshow(wordcloud, interpolation='bilinear')
    #plt.axis("off")
    if visual_ == 'show':
        plt.show()
    plt.savefig("out/" + class_+"wc.png",format='png')

    # How many features (x) are occurring how many times (y)? 
    '''fig2 = plt.figure(class_+"fc")
    fig2.suptitle(class_ + "- How many features (x) are occurring how many times (y)?")
    axes2 = fig2.gca()
    axes2.set_xlabel("Number of features")
    axes2.set_ylabel("Number of occurrences")
    fig2.savefig("out/" + class_+"fc.png",format='png')'''
    plt.title(class_ + "- How many features (x) are occurring how many times (y)?")
    plt.xlabel("Number of occurrences")
    plt.ylabel("Number of features")
    plt.hist(featcounts.values(), bins=200)
    #plt.box(on=False)
    if visual_ == 'show':
        plt.show()
    plt.savefig("out/" + class_+"fc.png",format='png')
    
    '''fig = plt.figure() # create figure
    ax = fig.add_subplot(nrows, ncols, plot_number) # create subplots
    '''
    '''fig = plt.figure(figsize=(7,7))
    gs = GridSpec(nrows=2, ncols=3, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0:3])
    #ax = sn.distplot(values, bins = 50, kde = False, ax = ax0, hist=True, hist_kws={"range": [0,2000]})
    ax = sn.distplot(values) #, kde = False, ax = ax0, hist=True, hist_kws={"range": [0,100]})
    ax.set(xlabel='Frequency of words', ylabel='Density', title ='Frequency of a word/feature')'''

    '''ax1 = fig.add_subplot(gs[1, 0])
    sn.barplot(x='categories',y='road',data=df,estimator=sum, ax = ax1)
    for p in ax1.patches:
        ax1.annotate(str(p.get_height()), (p.get_x()+0.1, p.get_height()))

    ax2 = fig.add_subplot(gs[1, 1])
    sn.barplot(x='categories',y='sauce',data=df,estimator=sum, ax = ax2)
    for p in ax2.patches:
        ax2.annotate(str(p.get_height()), (p.get_x()+0.1, p.get_height()))

    ax3 = fig.add_subplot(gs[1, 2])
    sn.barplot(x='categories',y='food',data=df,estimator=sum,  ax= ax3)
    for p in ax3.patches:
        ax3.annotate(str(p.get_height()), (p.get_x()+0.1, p.get_height()))'''

    #fig.suptitle("Features")
    #plt.show()

def observefeatures(vectorizer, bag_of_words, categories):
    features = vectorizer.get_feature_names()
    print("Total number of features:", len(features),"\n", random.sample(features,10))
    #print("Features:", features)
    print("Vocabulary:", type(vectorizer.vocabulary_),len(vectorizer.vocabulary_))
    print("Bag of words:", type(bag_of_words),bag_of_words.shape)
    print("BOW/Features - Non-zero values: ", bag_of_words.getnnz(),
            "Density:", bag_of_words.getnnz()*100/(bag_of_words.shape[0]*bag_of_words.shape[1]))
    print(type(bag_of_words[0]))

    # features_counts: How many times a feature/word has occurred in a corpus?
    feature_counts = np.sum(bag_of_words.toarray(),axis=0)
    feature_counts_df = pd.DataFrame(dict(features=features, counts = feature_counts))
	# TODO: How many words occur only once? Should we ignore the rest?

    print("Number of rare words in the dictionary:", len(feature_counts_df[feature_counts_df.counts==1]))
    print("Most frequently occurring features:\n", 
                                        feature_counts_df.sort_values('counts', ascending=False)[0:100])
    print("Features and counts")
    feature_counts_df = feature_counts_df.sort_values('counts', ascending=False)
    f, c = [], []
    for index, row in feature_counts_df.iterrows():
        try:
            #print("--->", row['features'])   #, type(row['features']), "---",, "---", row['counts']
            f.append(row['features'])
            c.append(row['counts'])
        except UnicodeEncodeError:
            print()
            continue
    feature_counts_df1 = pd.DataFrame(data={'features': f, 'counts': c})
    feature_counts_df1.to_csv(os.getcwd() + "/out/" + class_ + "features.csv")
    
    feature_counts = dict(zip(f, c)) 

    bag_of_words_ds_df = pd.DataFrame(bag_of_words.todense())
    bag_of_words_ds_df.columns = features
    bag_of_words_ds_df['categories'] = categories
    bag_of_words_ds_df.to_csv(os.getcwd() + "/out/" + class_ + "bow.csv")
    print("bag_of_words_ds_df:", type(bag_of_words_ds_df),bag_of_words_ds_df.info(), bag_of_words_ds_df.shape)

    # How many times a particular word appears in documents belonging to a particular category?
    visualize(feature_counts, bag_of_words_ds_df)

outliers = {'The President did not comment':-1,
			'I lost the keys':-1,
			'The team won the game':-1,
			'Sara has two kids':-1,
			'The ball went off the court':-1,
			'They had the ball for the whole game':-1,
			'The show is over':-1
			}	
def train(train_bow, vectorizer,  validate_set, categories_v,outliers, class_, validate_names):
    print("Training the model - started")
	# one-class SVM
    from sklearn import svm
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(train_bow)

    from sklearn.pipeline import Pipeline
    import joblib
    vec_clf = Pipeline([('vectorizer', vectorizer), ('oneclasssvm_clf', clf)])
    print(os.getcwd() + "/class/" + class_ + "class.pkl")
    location = joblib.dump(vec_clf, os.getcwd() + "/class/" + class_ + "class.pkl" , compress=9)
    print(location)
    print("Training the model - completed")
    
	#clf1 = joblib.load( os.getcwd() + "/class/" + class_ + "class.pkl") # Using pickled classifier
    predicted = vec_clf.predict(validate_set)
    if validate_names == []:
        for doc, category in zip(validate_set, predicted):
            print('%s => %r' % (category, doc)) 
    else:
        for doc, category in zip(validate_names, predicted):
            print('%s => %r' % (category, doc)) 
            #print('%s' % (category)) 

    pred_outliers = vec_clf.predict(list(outliers.keys()))
    for doc, category in zip(list(outliers.keys()), pred_outliers):
	    print('%s => %r' % (category, doc)) 
    
    accuracy = np.mean(predicted == np.asarray(categories_v))
    accuracy_outliers = np.mean(pred_outliers == np.asarray(list(outliers.values())))
    print("Accuracy",accuracy, accuracy_outliers)

    '''import matplotlib.pyplot as plt
    import seaborn as sn
	from sklearn import metrics
	print(type(categories_v), type(categories_v[0]), categories_v[0])
	print(type(predicted), type(predicted[0]), predicted[0])
	print(metrics.classification_report(categories_v, predicted)) #, target_names=['RED']))
	cm = metrics.confusion_matrix(categories_v, predicted)
	print(type(cm), cm)'''
    '''total1=sum(sum(cm))
	accuracy1=(cm[0,0]+cm[1,1])/total1
	sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
	specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
	print ('Accuracy : ', accuracy1, 'Sensitivity : ', sensitivity1,'Specificity : ', specificity1)'''
    '''sn.heatmap(cm, annot=True, fmt='.2f')
	plt.show()'''

def getvectorizer(transformer):
    if transformer == 'count':
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(tokenizer=tokenizer)
    elif transformer == 'tfidf':
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(tokenizer=tokenizer,max_features=features_)
    elif transformer == 'word2vec':
        from gensim.sklearn_api import D2VTransformer
        vectorizer = D2VTransformer(dm=1, size=50, min_count=2, iter=10, seed=0)  
    return vectorizer

startDT = process.recordstarttime(sys.argv)
dataset_folder = sys.argv[1]
filetype = sys.argv[2]
tokentype = sys.argv[3]
transformer = sys.argv[4]
class_ = sys.argv[5]
features_ = int(sys.argv[6])
visual_ = sys.argv[7]

TRAIN_DIR = dataset_folder +'train/'
VALIDATE_DIR = dataset_folder +'validate/'

if tokentype == "text":
    tokenizer = text.my_tokenizer
elif tokentype == "value":
    tokenizer = text.typetokenizer

validate_names = []
if filetype == "unstructured":
    corpusd_t, size, authors, creationdates = text.getcorpus(TRAIN_DIR)
    corpusd_v, size, authors, creationdates = text.getcorpus(VALIDATE_DIR)
    size_t = len(corpusd_t); size_v = len(corpusd_v)
    categories = [1 for c in range(size_t)];     categories_t = categories
    categories = [1 for c in range(size_v)];     categories_v = categories
    corpus_t = list(corpusd_t.values());     corpus_v = list(corpusd_v.values())
    validate_names = list(corpusd_v.keys())
    if transformer == 'word2vec':
        temp = []
        for corpus in corpus_t:
            corpus = tokenizer(corpus.decode('utf-8'))
            temp.append(corpus)
        corpus_t = temp
        print("corpus_t:", corpus_t[0])
        temp = []
        for corpus in corpus_v:
            corpus = tokenizer(corpus.decode('utf-8'))
            temp.append(corpus)
        corpus_v = temp
        print("corpus_v:", corpus_v[0])
elif filetype == "structured":
    if transformer == 'word2vec':
        w2v = True
    else:
        w2v = False
    corpus_t, categories_t = process.process_csv(TRAIN_DIR,tokenizer, w2v)
    corpus_v, categories_v = process.process_csv(VALIDATE_DIR,tokenizer, w2v)
    print(len(categories_t), categories_t)
    print(len(categories_v), categories_v)

print("Training data:", type(corpus_t), len(corpus_t),"Validation data:", type(corpus_v), len(corpus_v))

print("Processed the corpus")
vectorizer = getvectorizer(transformer)
print("getvectorizer - completed")
train_bow = vectorizer.fit_transform(corpus_t)
'''if transformer == 'word2vec':
    features = vectorizer.get_feature_names()
    print("\nFeatures", len(features), type(features), features)'''
if transformer != 'word2vec':
    observefeatures(vectorizer, train_bow, categories_t)
train(train_bow,vectorizer,corpus_v,categories_v,outliers, class_, validate_names)
process.recordstoptime(startDT)