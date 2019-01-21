import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize
import io
from statistics import *
from sklearn.feature_extraction.text import CountVectorizer


def find_max_mode(list1):
    list_table = statistics._counts(list1)
    len_table = len(list_table)

    if len_table == 1:
        max_mode = statistics.mode(list1)
    else:
        new_list = []
        for i in range(len_table):
            new_list.append(list_table[i][0])
        max_mode = max(new_list) # use the max value here
    return max_mode

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
            if len(votes)>1:
            	m = find_max_mode()
            	return m
            else:
            	return mode(votes)	

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


documents_f = open("pickled_algos/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()

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
random.shuffle(featuresets)

training_set = featuresets[:2900] 
testing_set = featuresets[2900:]

naivebayes_classifier_f = open("pickled_algos/naivebayes.pickle", "rb")
classifier = pickle.load(naivebayes_classifier_f)
naivebayes_classifier_f.close()

BNB_classifier_f = open("pickled_algos/BNB_classifier.pickle", "rb")
BNB_classifier = pickle.load(BNB_classifier_f)
BNB_classifier_f.close()

LogisticRegression_classifier_f = open("pickled_algos/LogisticRegression_classifier.pickle", "rb")
LogisticRegression_classifier = pickle.load(LogisticRegression_classifier_f)
LogisticRegression_classifier_f.close()

SGDClassifier_f = open("pickled_algos/SGDClassifier.pickle", "rb")
SGDClassifier = pickle.load(SGDClassifier_f)
SGDClassifier_f.close()

LinearSVC_classifier_f = open("pickled_algos/LinearSVC_classifier.pickle", "rb")
LinearSVC_classifier = pickle.load(LinearSVC_classifier_f)
LinearSVC_classifier_f.close()
'''
NuSVC_classifier_f = open("pickled_algos/NuSVC_classifier.pickle", "rb")
NuSVC_classifier = pickle.load(NuSVC_classifier_f)
NuSVC_classifier_f.close()
'''
voted_classifier = VoteClassifier(classifier,
                                  BNB_classifier,
                                  LinearSVC_classifier,
                                  SGDClassifier,
                                  LogisticRegression_classifier)

def sentiment(text):
    feats = find_features(text)
    classification = voted_classifier.classify(feats)
    classifier_conf = voted_classifier.confidence(feats)
    
    if classification == '1' and classifier_conf > 0.6:
        return "Positive Sentence" + " "+ voted_classifier.classify(feats),voted_classifier.confidence(feats)
    elif classification == '0' and classifier_conf >= 0.6:
        return "Negative Sentence" + " " +voted_classifier.classify(feats),voted_classifier.confidence(feats)
    else:
        return "Neutral" + " " + voted_classifier.classify(feats),voted_classifier.confidence(feats)
    
    #return voted_classifier.classify(feats),voted_classifier.confidence(feats)


#