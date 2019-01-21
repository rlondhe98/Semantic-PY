from textblob import TextBlob
from nltk.corpus import movie_reviews
import nltk
from nltk.corpus import stopwords
import random
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from nltk.classify import ClassifierI
from textblob.classifiers import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from sklearn.svm import SVC, LinearSVC, NuSVC
stop_words = set(stopwords.words("english"))

documents_f = open("pickled_algos/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()
random.shuffle(documents)

word_features_f = open("pickled_algos/word_features.pickle", "rb")
word_features = pickle.load(word_features_f)
word_features_f.close()

def find_features(document):
    words = word_tokenize(document)                                 
    features = {}
    for w in word_features:
        features[w] = (w in words)                          

    return features

featuresets_f = open("pickled_algos/featuresets.pickle", "rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()

train = featuresets[0:6500]
test = featuresets[6500:0]

naivebayes_classifier_f = open("pickled_algos/naivebayes.pickle", "rb")
c1 = pickle.load(naivebayes_classifier_f)
naivebayes_classifier_f.close()

MNB_classifier_f = open("pickled_algos/MNB_classifier.pickle", "rb")
c2 = pickle.load(MNB_classifier_f)
MNB_classifier_f.close()

BNB_classifier_f = open("pickled_algos/BNB_classifier.pickle", "rb")
c3 = pickle.load(BNB_classifier_f)
BNB_classifier_f.close()

LogisticRegression_classifier_f = open("pickled_algos/LogisticRegression_classifier.pickle", "rb")
c4 = pickle.load(LogisticRegression_classifier_f)
LogisticRegression_classifier_f.close()

SGDClassifier_f = open("pickled_algos/SGDClassifier.pickle", "rb")
c5 = pickle.load(SGDClassifier_f)
SGDClassifier_f.close()

SVC_classifier_f = open("pickled_algos/SVC_classifier.pickle", "rb")
c6 = pickle.load(SVC_classifier_f)
SVC_classifier_f.close()

LinearSVC_classifier_f = open("pickled_algos/LinearSVC_classifier.pickle", "rb")
c7 = pickle.load(LinearSVC_classifier_f)
LinearSVC_classifier_f.close()

NuSVC_classifier_f = open("pickled_algos/NuSVC_classifier.pickle", "rb")
c8 = pickle.load(NuSVC_classifier_f)
NuSVC_classifier_f.close()

def analyzeSentiment(text):
	classifierList = [c1,c2,c3,c4,c5,c6,c7,c8]
	pol = 0
	chosen_classifier = 0
	for i in classifierList:
		analysis = TextBlob(text, classifier = i)
		if analysis.sentiment.polarity > pol:
			pol = analysis.sentiment.polarity
			chosen_classifier = i
	print(analysis.sentiment.polarity)
'''
	if analysis.sentiment.polarity >= 0.0001:
		if analysis.sentiment.polarity > 0:
		    return "Positive Sentence!!  " +"Confidence = "+str(analysis.sentiment.polarity*100)+"%"
	elif analysis.sentiment.polarity <= -0.0001:
		if analysis.sentiment.polarity <= 0:
		    return "Negative Sentence!! " + "Confidence = "+str(-analysis.sentiment.polarity*100)+"%"
	else:
		return "Neutral Sentence " + "Confidence = "+str(analysis.sentiment.polarity*100)+"%"'''