import sentimentanalysis as sa
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import PorterStemmer

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))
while True:
	text = input("Enter Your tweet.. ")

	temp = word_tokenize(text)
	finalOutput=[]

	for j in temp:
		if len(j)>1:
			finalOutput.append(j)
	sent = ""

	for i in range(len(finalOutput)):
		sent = sent + finalOutput[i] + " "							

	#print(ps.stem(sent))
	print(sa.analyzeSentiment(sent))