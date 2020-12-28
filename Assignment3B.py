"""This is a python program where we are reading the data from the Reuters Corpus 21578 new articles. Here we are reading the Topics, DateLine and topics of each article. We split the dataset into 80/20 where first 80% of the dataset is used to train the algorithm and other 20% of the dataset is used to test the algorithm. We have used same dataset split for each run and each algorithm to compare the performance using confusion matrix, Precision and Recall.

Resources:
Sam Scott's parse_reuters.py code to read the data and the labels
Sam Scott's text_classification.py code to parse the text and create bagOfWords

Prerak Patel, Student, Mohawk College, 2020
"""

import xml.etree.ElementTree as et
import numpy as np
import time
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from nltk.corpus import stopwords

## Reading Reuters Corpus
def read_reuters(path="Reuters21578", limit=21578):
    ''' Reads the Reuters-21578 corpus from the given path. (This is assumed to
    be the cleaned-up version of the corpus provided for this code.) The limit
    parameter can be used to stop reading after a certain number of documents
    have been read.'''
    def get_dtags(it, index):
        '''Helper function to parse the <D> elements'''
        dtags = []
        while it[index+1].tag == "D":
            dtags.append(it[index+1].text)
            index += 1
        return dtags, index

    docs = []
    numdocs = 0

    for i in range(22):
        pad = ""
        if i<10:
            pad = "0"

        tree = et.parse(path+'/reut2-0'+pad+str(i)+'.sgm')
        root = tree.getroot()

        it = list(tree.iter())


        index = 0
        while index < len(it):
            if it[index].tag == "REUTERS":
                if numdocs == limit:
                    return docs
                docs.append({})
                numdocs+=1
            elif it[index].tag.lower() in ["topics", "places","people","orgs","exchanges","companies"]:
                docs[numdocs-1][it[index].tag.lower()], index = get_dtags(it, index)
            elif numdocs > 0:
                docs[numdocs-1][it[index].tag.lower()] = it[index].text

            index +=1

    return docs

# Getting labels array
def get_labels(docs, labeltype,labelname):
    # labels array to store the class data
    labels = []
    # looping through each article document
    for doc in docs:
        # check if the topics exists in the article
        if len(doc[labeltype]) == 0:
            labels.append(0)
        # if topcis exists
        else:
            # keeping track number of labels in the article
            labelCounter = len(doc[labeltype])
            # iterating through each topic in article
            for label in doc[labeltype]:
                # check if we found topic we are looking for
                if label == labelname:
                    labels.append(1)
                    break;
                else:
                    # if we are at the last topic in topcis list
                    if labelCounter == 1:
                        labels.append(0)
                        break;
                    # increase the counter to check for the next topic in the list
                    else:
                        labelCounter -= 1

    # return labels array
    return labels

# reading the data returned from the Reuters corpus
def get_body(docs):
    # array to store text related to the article
    body_data = []

    # iterate through each article in the document
    for doc in docs:
        # try and except for the parsing the text
        try:
            # reading the text for the title, dateline & body and parsing the text
            each_doc = textParse(doc["title"] + doc["dateline"] + doc["body"])
            body_data.append(each_doc)
        except:
            body_data.append("")

    # returning the array
    return body_data

# creating the vocabulary from the dataset
def createVocabList(dataSet):
    """ dataSet is a list of word lists. Returns the set of words in the dataSet
    as a list."""
    # declaring as set to store only unqiue words
    vocabSet = set()  #create empty set
    for document in dataSet:
        # union of the two sets
        vocabSet = vocabSet.union(set(document))
    # returning the vocabulary list
    return list(vocabSet)

# parsing the text data recieved
def textParse(text):
    """ A utility to split a string into a list of lowercase words, removing punctuation"""
    import string
    return [word.lower().strip(string.punctuation) for word in text.split()]

# stemming words to the root word
def stemWords(list):

    # iterating through the list recieved
    for index,vocab in enumerate(list):
        # looping through 3 times with values 1, 2, 3
        for x in range(1,4):
            # slicing the last two letters from the word to check if includes "ed" or "er"
            if(vocab[-x:] == "ed" or vocab[-x:] == "er"):
                # adding one to the x value to keep "e" when slicing
                slice = x - 1
                list[index] = vocab[:-slice]

            # slicing the last two letters from the word to check if includes "ly"
            if(vocab[-x:] == "ly"):
                # slice the word and add it to the list
                list[index] = vocab[:-x]

            # slicing the last three letters from the word to check if includes "ing"
            if(vocab[-x:] == "ing"):
                slice = x - 3
                # slice the word and add it to the list
                list[index] = vocab[:-x]

            # slicing the last one letter from the word to check if includes "ly"
            if(vocab[-x:] == "s"):
                # adding 1 to the x value to check for the 's before checking for just "s"
                slice = x + 1
                # checking if the word includes 's
                if vocab[-slice:] == "'s":
                    # slice the word and add it to the list
                    list[index] = vocab[:-slice]

                else:
                    # slice the word and add it to the list
                    list[index] = vocab[:-x]

    # iterating through list
    for index,vocab in enumerate(list):
        # looping through 2 times with value 2 and 3
        for x in range(2,4):
            # check if last two letters includes "un", "re" or "de"
            if(vocab[:x] == "un" or vocab[:x] == "re" or vocab[:x] == "de"):
                # slice the word and add it to the list
                list[index] = vocab[x:]

            # check if last three letters includes "dis"  or "mis"
            if(vocab[:x] == "dis" or vocab[:x] == "mis"):
                # slice the word and add it to the list
                list[index] = vocab[x:]

    # return stemmed list
    return list

# function to check occurences of the word in the inputlist by returing vocablist with integer array representing the repeatation of the word
def bagOfWords(vocabList, inputList):
    """ vocabList is a set of words (as a list). inputList is a list of words
    occurring in a document. Returns a list of integers indicating how many
    times each word in the vocabList occurs in the inputList"""
    d = {}
    for word in inputList:
        d[word] = d.get(word,0)+1
    bagofwords = []
    for word in vocabList:
        bagofwords.append(d.get(word,0))
    return bagofwords

# function to check occurences of the word in the inputlist by returing vocablist with integer array representing by 0 or 1
def setOfWords(vocabList, inputList):
    """ vocabList is a set of words (as a list). inputList is a list of words
    occurring in a document. Returns a list of 1's and 0's to indicate
    the presence or absence of each word in vocabList"""
    d = {}
    for word in inputList:
        d[word] = 1
    setofwords = []
    for word in vocabList:
        setofwords.append(d.get(word,0))
    return setofwords

# function to calculate the confusion matrix by accepting two arguments test_labels and test_pred_labels which represents actual array for testing labels and predicted array of test labels respectively
def confusionMatrix(test_labels,test_pred_labels):
    # converting it to numpy array to the slice the array for the calculations
    test_labels = np.array(test_labels)
    test_pred_labels = np.array(test_pred_labels)

    # test_labels_0 is the array with indexes where the test_labels is 0
    test_labels_0 = np.where(test_labels == 0)[0]
    # test_labels_1 is the array with indexes where the test_labels is 1
    test_labels_1 = np.where(test_labels == 1)[0]
    # test_pred_0 is the array with indexes where the test_pred_labels is 0
    test_pred_0 = np.where(test_pred_labels == 0)[0]
    # test_pred_1 is the array with indexes where the test_pred_labels is 1
    test_pred_1 = np.where(test_pred_labels == 1)[0]

    # pred_1_true_1 indicates when the test_pred_labels predicted 1 and it was true
    pred_1_true_1 = 0
    # pred_1_true_0 indicates when the test_pred_labels predicted 1 and it was false
    pred_1_true_0 = 0
    # pred_0_true_0 indicates when the test_pred_labels predicted 0 and it was true
    pred_0_true_0 = 0
    # pred_0_true_1 indicates when the test_pred_labels predicted 0 and it was false
    pred_0_true_1 = 0

    # iterate through the test_pred_0 array
    for test_pred_0_label in test_pred_0:
        # check if the current index exists in test_labels_0 to confirm the prediction was true
        if test_pred_0_label in test_labels_0:
            pred_0_true_0 += 1
        # if the prediction was false increase the counter for pred_0_true_1
        else:
            pred_0_true_1 += 1

    # iterate through the test_pred_1 array
    for test_pred_1_label in test_pred_1:
        # check if the current index exists in test_labels_0 to confirm the prediction was true
        if test_pred_1_label in test_labels_1:
            pred_1_true_1 += 1
        # if the prediction was false increase the counter for pred_1_true_0
        else:
            pred_1_true_0 += 1

    # returning the values prediction counters
    return [pred_0_true_0,pred_0_true_1,pred_1_true_1,pred_1_true_0]

# function to make the calculation of accuracy, precision and recall
def calculation(pred_0_true_0,pred_0_true_1,pred_1_true_1,pred_1_true_0):
    accuracyResult = (pred_1_true_1 + pred_0_true_0)/(pred_1_true_1 + pred_0_true_1 + pred_1_true_0 + pred_0_true_0)
    precision = (pred_1_true_1)/(pred_1_true_1 + pred_1_true_0)
    recall = (pred_1_true_1)/(pred_1_true_1 + pred_0_true_1)

    # returning the value for the accuracy, precision and recall
    return accuracyResult, precision, recall

# function to print the combined Confusion Matrix
def combinedConfusionMatrix(confusionMatrixArray):
    # converting the array into numpy array for slicing
    confusionMatrixArray = np.array(confusionMatrixArray)
    # calling the function for the calculation
    accuracyResult, precision, recall =  calculation(confusionMatrixArray[:,0].mean(),confusionMatrixArray[:,1].mean(),confusionMatrixArray[:,2].mean(),confusionMatrixArray[:,3].mean())

    print("\n\t\t\tConfusion Matrix\n\t" + "="*30 +"\n\t\tPredicted 1\t\tPredicted 0\nTrue 1\t\t" + str(np.round(confusionMatrixArray[:,2].mean(),2)) + "\t\t\t" + str(np.round(confusionMatrixArray[:,1].mean(),2)) + "\nTrue 0\t\t" + str(np.round(confusionMatrixArray[:,3].mean(),2)) + "\t\t\t\t" + str(np.round(confusionMatrixArray[:,0].mean(),2)))

    print("-"*36 +"\nAccuracy: " + str(np.round(accuracyResult,2)) + " * 100 = " + str(np.round(accuracyResult*100,2)) + "\nPrecision: " + str(np.round(precision,2)) + "\nRecall: " + str(np.round(recall,2)) + "\n" + "-"*36)

# creating 80/20 training and testing split
def createDataSet(data_set,data_set_labels):
    # training data
    train_data = data_set[:int(len(data_set)*0.8)]
    train_labels = data_set_labels[:int(len(data_set)*0.8)]
    # testing data
    test_data = data_set[int(len(data_set)*0.8):]
    test_labels = data_set_labels[int(len(data_set)*0.8):]

    # returning training and testing data
    return train_data, train_labels, test_data, test_labels

# function for multinomial prediction
def multinomial(data_set,data_set_labels):
    train_data, train_labels, test_data, test_labels = createDataSet(data_set,data_set_labels)

    # training
    nb.fit(train_data, train_labels)

    # prediction
    test_pred_labels = nb.predict(test_data)

    # confusion-matrix calculation
    confusionMatrixCalc = confusionMatrix(test_labels,test_pred_labels)

    # returning the calculation
    return confusionMatrixCalc

# function for complement prediction
def complement(data_set,data_set_labels):
    train_data, train_labels, test_data, test_labels = createDataSet(data_set,data_set_labels)

    # training
    cnb.fit(train_data, train_labels)

    # prediction
    test_pred_labels = cnb.predict(test_data)

    # confusion-matrix calculation
    confusionMatrixCalc = confusionMatrix(test_labels,test_pred_labels)

    # returning the calculation
    return confusionMatrixCalc

# function for bernoulli prediction
def bernoulli(data_set,data_set_labels):
    train_data, train_labels, test_data, test_labels = createDataSet(data_set,data_set_labels)

    # training
    bnb.fit(train_data, train_labels)

    # prediction
    test_pred_labels = bnb.predict(test_data)

    # confusion-matrix calculation
    confusionMatrixCalc = confusionMatrix(test_labels,test_pred_labels)

    # returning the calculation
    return confusionMatrixCalc

# function to return the array after removing all the stop words from the list
def stopWords(docsData):
    # using the nltk library to get the stop words of English language
    stop_words = set(stopwords.words('english'))
    # array to store filtered list
    filteredDocs = []

    # iterating through the list
    for docData in docsData:
        # check if the word from the list exists in the stop words list
        if docData not in stop_words:
            filteredDocs.append(docData)

    # returning filtered list
    return filteredDocs

## Running the Program
print("Reading...")
# calling the read Reuters function
docs = read_reuters("Reuters21578",500)
# getting the "topics", "dateline" and "body" from the docs
docs_data = get_body(docs)

print("Loading...")
# creating vocabulary list from the docs_data
vocab_list = createVocabList(docs_data)

# 5 classifiers stored in an array
task_labels = ["earn", "acq", "money-fx", "crude", "grain"]
nb = MultinomialNB()
bnb = BernoulliNB()
cnb = ComplementNB()

## Bag of words Multinomial NB

# array to store confusion matrix calculation
bagOfWordsMultinomialArray = []

for task_label in task_labels:
    data_set_labels = get_labels(docs,"topics",task_label)
    data_set = []
    for eachDoc in docs_data:
        data_set.append(bagOfWords(vocab_list, eachDoc))

    # passing data set and target labels to train and test the algorithm
    bagOfWordsMultinomialResults = multinomial(data_set,data_set_labels)
    # appending the confusion matrix result produced to the array
    bagOfWordsMultinomialArray.append(bagOfWordsMultinomialResults)

print("Done...\n\n\t\tBag of Words Multinomial")
# Combined Confusion-matrix Calculation
combinedConfusionMatrix(bagOfWordsMultinomialArray)

## Bag of words Complement NB
print("Loading...")
# array to store confusion matrix calculation
bagOfWordsComplementArray = []

for task_label in task_labels:

    data_set_labels = get_labels(docs,"topics",task_label)
    data_set = []
    for eachDoc in docs_data:
        data_set.append(bagOfWords(vocab_list, eachDoc))

    # passing data set and target labels to train and test the algorithm
    bagOfWordsComplementResults = complement(data_set,data_set_labels)
    # appending the confusion matrix result produced to the array
    bagOfWordsComplementArray.append(bagOfWordsComplementResults)

print("Done...\n\n\t\tBag of Words Complement")
# Combined Confusion-matrix Calculation
combinedConfusionMatrix(bagOfWordsComplementArray)

## Bag of stems(Stemming) Multinomial NB
print("Loading...")
# array to store confusion matrix calculation
bagOfStemsMultinomialArray = []

for task_label in task_labels:

    data_set_labels = get_labels(docs,"topics",task_label)
    # stemWords method call to stem to the root word
    vocab_list_stemmed = stemWords(vocab_list)
    data_set = []
    for eachDoc in docs_data:
        # stemWords method call to stem to the root word
        eachDoc_stemmed = stemWords(eachDoc)
        data_set.append(bagOfWords(vocab_list_stemmed, eachDoc_stemmed))

    # passing data set and target labels to train and test the algorithm
    bagOfStemsMultinomialResults = multinomial(data_set,data_set_labels)
    # appending the confusion matrix result produced to the array
    bagOfStemsMultinomialArray.append(bagOfStemsMultinomialResults)


print("Done...\n\n\t\tBag of Stems Multinomial")
# Combined Confusion-matrix Calculation
combinedConfusionMatrix(bagOfStemsMultinomialArray)

## Bag of stems(Stemming) Complement NB
print("Loading...")
# array to store confusion matrix calculation
bagOfStemsComplementArray = []

for task_label in task_labels:

    data_set_labels = get_labels(docs,"topics",task_label)
    # stemWords method call to stem to the root word
    vocab_list_stemmed = stemWords(vocab_list)
    data_set = []
    for eachDoc in docs_data:
        # stemWords method call to stem to the root word
        eachDoc_stemmed = stemWords(eachDoc)
        data_set.append(bagOfWords(vocab_list_stemmed, eachDoc_stemmed))

    # passing data set and target labels to train and test the algorithm
    bagOfStemsComplementResults = complement(data_set,data_set_labels)
    # appending the confusion matrix result produced to the array
    bagOfStemsComplementArray.append(bagOfStemsComplementResults)


print("Done...\n\n\t\tBag of Stems Complement")
# Combined Confusion-matrix Calculation
combinedConfusionMatrix(bagOfStemsComplementArray)


## Bag of words w/ STOP WORDS Multinomial NB
print("Loading...")
# array to store confusion matrix calculation
bagOfWordsMultinomialArray = []

for task_label in task_labels:

    data_set_labels = get_labels(docs,"topics",task_label)
    stop_words_vocab_list = stopWords(vocab_list)
    data_set = []
    for eachDoc in docs_data:
        # stopWords method call to remove unnecessary words with no actual meaning
        eachDoc_stop_words = stopWords(eachDoc)
        data_set.append(bagOfWords(stop_words_vocab_list, eachDoc_stop_words))

    # passing data set and target labels to train and test the algorithm
    bagOfWordsMultinomialResults = multinomial(data_set,data_set_labels)
    # appending the confusion matrix result produced to the array
    bagOfWordsMultinomialArray.append(bagOfWordsMultinomialResults)

print("Done...\n\n\t\tBag of Words w/ STOP WORDS Multinomial")
# Combined Confusion-matrix Calculation
combinedConfusionMatrix(bagOfWordsMultinomialArray)

## Bag of words w/ STOP WORDS Complement NB
print("Loading...")
# array to store confusion matrix calculation
bagOfWordsComplementArray = []

for task_label in task_labels:

    data_set_labels = get_labels(docs,"topics",task_label)
    stop_words_vocab_list = stopWords(vocab_list)
    data_set = []
    for eachDoc in docs_data:
        # stopWords method call to remove unnecessary words with no actual meaning
        eachDoc_stop_words = stopWords(eachDoc)
        data_set.append(bagOfWords(stop_words_vocab_list, eachDoc_stop_words))

    # passing data set and target labels to train and test the algorithm
    bagOfWordsComplementResults = complement(data_set,data_set_labels)
    # appending the confusion matrix result produced to the array
    bagOfWordsComplementArray.append(bagOfWordsComplementResults)

print("Done...\n\n\t\tBag of Words w/ STOP WORDS Complement")
# Combined Confusion-matrix Calculation
combinedConfusionMatrix(bagOfWordsComplementArray)

## Bag of stems(Stemming) w/ STOP WORDS Multinomial NB
print("Loading...")
# array to store confusion matrix calculation
bagOfStemsMultinomialArray = []

for task_label in task_labels:

    data_set_labels = get_labels(docs,"topics",task_label)
    # stopWords method call to remove unnecessary words with no actual meaning
    stop_words_vocab_list = stopWords(vocab_list)
    # stemWords method call to stem to the root word
    vocab_list_stemmed = stemWords(stop_words_vocab_list)
    data_set = []
    for eachDoc in docs_data:
        # stopWords method call to remove unnecessary words with no actual meaning
        eachDoc_stop_words = stopWords(eachDoc)
        # stemWords method call to stem to the root word
        eachDoc_stemmed = stemWords(eachDoc_stop_words)
        data_set.append(bagOfWords(vocab_list_stemmed, eachDoc_stemmed))

    # passing data set and target labels to train and test the algorithm
    bagOfStemsMultinomialResults = multinomial(data_set,data_set_labels)
    # appending the confusion matrix result produced to the array
    bagOfStemsMultinomialArray.append(bagOfStemsMultinomialResults)

print("Done...\n\n\t\tBag of Stems w/ STOP WORDS Multinomial")
# Combined Confusion-matrix Calculation
combinedConfusionMatrix(bagOfStemsMultinomialArray)

## Bag of stems(Stemming) w/ STOP WORDS Complement NB
print("Loading...")
# array to store confusion matrix calculation
bagOfStemsComplementArray = []

for task_label in task_labels:

    data_set_labels = get_labels(docs,"topics",task_label)
    # stopWords method call to remove unnecessary words with no actual meaning
    stop_words_vocab_list = stopWords(vocab_list)
    # stemWords method call to stem to the root word
    vocab_list_stemmed = stemWords(stop_words_vocab_list)
    data_set = []
    for eachDoc in docs_data:
        # stopWords method call to remove unnecessary words with no actual meaning
        eachDoc_stop_words = stopWords(eachDoc)
        # stemWords method call to stem to the root word
        eachDoc_stemmed = stemWords(eachDoc_stop_words)
        data_set.append(bagOfWords(vocab_list_stemmed, eachDoc_stemmed))

    # passing data set and target labels to train and test the algorithm
    bagOfStemsComplementResults = complement(data_set,data_set_labels)
    # appending the confusion matrix result produced to the array
    bagOfStemsComplementArray.append(bagOfStemsComplementResults)


print("Done...\n\n\t\tBag of Stems w/ STOP WORDS Complement")
# Combined Confusion-matrix Calculation
combinedConfusionMatrix(bagOfStemsComplementArray)


## Set of words Bernoulli NB
print("Loading...")
# array to store confusion matrix calculation
setOfWordsMultinomialArray = []

for task_label in task_labels:
    data_set_labels = get_labels(docs,"topics",task_label)
    data_set = []
    for eachDoc in docs_data:
        data_set.append(setOfWords(vocab_list, eachDoc))

    # passing data set and target labels to train and test the algorithm
    setOfWordsMultinomialResults = bernoulli(data_set,data_set_labels)
    # appending the confusion matrix result produced to the array
    setOfWordsMultinomialArray.append(setOfWordsMultinomialResults)

print("Done...\n\n\t\tSet of Words Bernoulli")
# Combined Confusion-matrix Calculation
combinedConfusionMatrix(setOfWordsMultinomialArray)




