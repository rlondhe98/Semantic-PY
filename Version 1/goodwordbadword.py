import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize, RegexpTokenizer
import io
from statistics import *
from nltk.corpus import stopwords

tokenizer = RegexpTokenizer(r'\w+')

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


with open("goodandbadsentences.txt","r") as text_file:
    documents = text_file.read().split('\n')

documents = list((line.split("\t") for line in documents if len(line.split("\t"))==2 and line.split("\t")[1]!=''))

#documents
save_documents = open("pickled_algos/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()

stop_words = set(stopwords.words("english"))

all_words = []
for i in documents:	
	temp = tokenizer.tokenize(i[0].lower())
	for j in temp:
		if not j.isdigit() and len(j)>2:
			if j not in stop_words:
				all_words.append(j)
#print(all_words[0:10])
all_words = nltk.FreqDist(all_words)
#print(all_words.most_common(10))

word_features = []#list(all_words.keys())[:500]
for i in all_words.most_common(1000):
	word_features.append(i[0])

save_word_features = open("pickled_algos/word_features.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()

def find_features(document):
	words = set(document)									
	features = {}
	for w in word_features:
		features[w] = (w in words)							

	return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]
random.shuffle(featuresets)
#print(featuresets[0:3])
#featuresets pickle
save_featuresets = open("pickled_algos/featuresets.pickle","wb")
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()

training_set = featuresets[:6500] 
#print(training_set[0:10])
testing_set = featuresets[6500:]

#Classifier testing
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes algo accuracy percent: ", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features()

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("BNB classifier accuracy percent: ", (nltk.classify.accuracy(BNB_classifier, testing_set))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent: ", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier = SklearnClassifier(SGDClassifier())
SGDClassifier.train(training_set)
print("SGDClassifier accuracy percent: ", (nltk.classify.accuracy(SGDClassifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent: ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

'''
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent: ", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
'''
voted_classifier = VoteClassifier(classifier, BNB_classifier, LogisticRegression_classifier, SGDClassifier, LinearSVC_classifier)
print("voted_classifier accuracy percent: ", (nltk.classify.accuracy(voted_classifier, testing_set))*100)
print("Classification: ", voted_classifier.classify(testing_set[0][0]), "confidence %: ", voted_classifier.confidence(testing_set[0][0]))


#original Naive bayes classifier
save_classifier_original_naivebayes = open("pickled_algos/naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier_original_naivebayes)
save_classifier_original_naivebayes.close()

#BNB Naive bayes classifier
save_BNB_classifier = open("pickled_algos/BNB_classifier.pickle","wb")
pickle.dump(BNB_classifier, save_BNB_classifier)
save_BNB_classifier.close()


#LogisticRegression_classifier
save_LogisticRegression_classifier = open("pickled_algos/LogisticRegression_classifier.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_LogisticRegression_classifier)
save_LogisticRegression_classifier.close()


#SGDClassifier
save_SGDClassifier = open("pickled_algos/SGDClassifier.pickle","wb")
pickle.dump(SGDClassifier, save_SGDClassifier)
save_SGDClassifier.close()

#LinearSVC_classifier
save_LinearSVC_classifier = open("pickled_algos/LinearSVC_classifier.pickle","wb")
pickle.dump(LinearSVC_classifier, save_LinearSVC_classifier)
save_LinearSVC_classifier.close()

'''
#NuSVC_classifier
save_NuSVC_classifier = open("pickled_algos/NuSVC_classifier.pickle","wb")
pickle.dump(NuSVC_classifier, save_NuSVC_classifier)
save_NuSVC_classifier.close()
'''