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
random.shuffle(documents)
#print(documents[0:3])
#documents
save_documents = open("pickled_algos/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()

allwords = []
for i in documents:
	for j in word_tokenize(i[0]):
		if j not in stop_words and len(j) > 2 and type(j) != 'Int':
			allwords.append(j.lower())

allwords = nltk.FreqDist(allwords)
#print(len(allwords))
#print(allwords.most_common(10))

word_features = []
for i in allwords.most_common(1000):
	word_features.append(i[0])
#print(len(word_features))
#print(word_features[0:10])
save_word_features = open("pickled_algos/word_features.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()

def find_features(document):
	words = set(document)									#2
	features = {}
	for w in word_features:
		features[w] = (w in words)							#3

	return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]
#print(featuresets[1])
#print(len(featuresets))
save_featuresets = open("pickled_algos/featuresets.pickle","wb")
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()

train = featuresets[0:6500]
test = featuresets[6500:0]

#Bringing in the classifiers
c1 = NaiveBayesClassifier(train)

c2 = SklearnClassifier(MultinomialNB())
c2.train(train)

c3 = SklearnClassifier(BernoulliNB())
c3.train(train)

c4 = SklearnClassifier(LogisticRegression())
c4.train(train)

c5 = SklearnClassifier(SGDClassifier())
c5.train(train)

c6 = SklearnClassifier(SVC())
c6.train(train)

c7 = SklearnClassifier(LinearSVC())
c7.train(train)

c8 = SklearnClassifier(NuSVC())
c8.train(train)

#c7 = VoteClassifier(classifier, BNB_classifier, LogisticRegression_classifier, SGDClassifier, LinearSVC_classifier)
#print("voted_classifier accuracy percent: ", (nltk.classify.accuracy(voted_classifier, testing_set))*100)
#print("Classification: ", voted_classifier.classify(testing_set[0][0]), "confidence %: ", voted_classifier.confidence(testing_set[0][0]))

#Start Predicting


analysis = TextBlob("hey there awesome person in the world", classifier = c1)

if analysis.sentiment.polarity >= 0.5:
	if analysis.sentiment.polarity > 0:
	    print("Positive Sentence!!  " +"Confidence = "+str(analysis.sentiment.polarity*100)+"%")
elif analysis.sentiment.polarity <= -0.5:
	if analysis.sentiment.polarity <= 0:
	    print("Negative Sentence!! " + "Confidence = "+str(analysis.sentiment.polarity*100)+"%")
else:
	print("Neutral Sentence " + "Confidence = "+str(analysis.sentiment.polarity*100)+"%")

#original Naive bayes classifier
save_classifier_original_naivebayes = open("pickled_algos/naivebayes.pickle","wb")
pickle.dump(c1, save_classifier_original_naivebayes)
save_classifier_original_naivebayes.close()

#MNB Naive bayes classifier
save_MNB_classifier = open("pickled_algos/MNB_classifier.pickle","wb")
pickle.dump(c2, save_MNB_classifier)
save_MNB_classifier.close()

#BNB Naive bayes classifier
save_BNB_classifier = open("pickled_algos/BNB_classifier.pickle","wb")
pickle.dump(c3, save_BNB_classifier)
save_BNB_classifier.close()

#LogisticRegression_classifier
save_LogisticRegression_classifier = open("pickled_algos/LogisticRegression_classifier.pickle","wb")
pickle.dump(c4, save_LogisticRegression_classifier)
save_LogisticRegression_classifier.close()

#SGDClassifier
save_SGDClassifier = open("pickled_algos/SGDClassifier.pickle","wb")
pickle.dump(c5, save_SGDClassifier)
save_SGDClassifier.close()

#SVC_classifier
save_SVC_classifier = open("pickled_algos/SVC_classifier.pickle","wb")
pickle.dump(c6, save_SVC_classifier)
save_SVC_classifier.close()


#LinearSVC_classifier
save_LinearSVC_classifier = open("pickled_algos/LinearSVC_classifier.pickle","wb")
pickle.dump(c7, save_LinearSVC_classifier)
save_LinearSVC_classifier.close()


#NuSVC_classifier
save_NuSVC_classifier = open("pickled_algos/NuSVC_classifier.pickle","wb")
pickle.dump(c8, save_NuSVC_classifier)
save_NuSVC_classifier.close()


'''

while True:
	analysis = TextBlob(input('Enter Your Tweet: '))

	print(analysis.sentiment.polarity)

	if analysis.sentiment.polarity > 0:
		print("Positive Sentence")
	elif analysis.sentiment.polarity < 0:
		print("Negative Sentence")

pos_count = 0
pos_correct = 0

with open("positive.txt","r", encoding='latin-1') as f:
    for line in f.read().split('\n'):
        analysis = TextBlob(line)

        if analysis.sentiment.polarity >= 0.0001:
            if analysis.sentiment.polarity > 0:
                pos_correct += 1
            pos_count +=1


neg_count = 0
neg_correct = 0

with open("negative.txt","r", encoding='latin-1') as f:
    for line in f.read().split('\n'):
        analysis = TextBlob(line)
        if analysis.sentiment.polarity <= -0.0001:
            if analysis.sentiment.polarity <= 0:
                neg_correct += 1
            neg_count +=1

print("Positive accuracy = {}% via {} samples".format(pos_correct/pos_count*100.0, pos_count))
print("Negative accuracy = {}% via {} samples".format(neg_correct/neg_count*100.0, neg_count))
'''