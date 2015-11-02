"""
Performing NLP using scikit-learn and NLTK. 

by Patricia Decker, 11/01/2015, Hackbright Academy Independent Project

Data Structure: Yelp JSON

review_dict = {'type': 'review',
               'business_id': rest_name,
               'user_id': 'greyhoundmama',
               'stars': rest_stars,
               'text': review_text,
               'date': yelp_date,
               'votes': vote_dict
               'target': default=None
               }

The classifier will be classifying on review_dict['text'], the review.
target

Two-step classification process to perform sentiment analysis of 
restaurant reviews to determine whether a restaurant is accommdating of
food allergies.

First, classify the documents (reviews) by category, namely whether it is
relevant to a particular allergy or not. A document may have many categories.
Can use process similar to 'scikit-learn.py' using a bag of words approach to
categorize the documents, as the presence of the word is more important than
the position of the word for this classification.

Second, perform sentiment analysis of a particular review and determine a 
probability that the reviewer's experience *specifically related to the 
particular food allergy* was positive. Based on this probability, the
analyzed review will be classified on a good-to-bad spectrum:

1.0-0.8 = Excellent
0.8-0.6 = Good
0.6-0.4 = Neutral
0.4-0.2 = Shady
0.2-0.0 = Bad
(depends on the accuracy of the prediction, if it can be this granular)

... Individual reviews will be assigned a probability.

... TODO: What is the probability that a restaurant is positive or negative
    for a particular condition based on multiple relevant reviews?



LinearSVC classifier that takes features vectors consisting of tokenized
reviews that have been converted to numerical values (counts) and
transformed to account for term frequency and inverse document frequency
(tf-idf). Tested on toy data set: 45 hand-labeled reviews that, for the
most part, already contain the word 'gluten'.
"""
import random
import numpy as np
from sklearn.datasets import base as sk_base
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline

# import reviewfilter as rf


#### LOAD DATA ########################################################

# directory containing toy data set: reviews by pdecks as .txt files
# must be preprocessed with 'preprocess-reviews.py' if the .txt files
# contain more than just review information delimited on pipes
container_path = 'pdecks-reviews/'

categories = ['bad', 'excellent', 'good', 'limited', 'neutral', 'shady']

# load the list of files matching the categories
# BEWARE OF LURKING .DS_Store files!! those are not 'utf-8'
# and will throw a UnicodeDecodeError
pdecks_reviews = sk_base.load_files(container_path,
                                  categories=categories,
                                  encoding='utf-8')

# Split the dataset into a test set and a training set
X = np.array(pdecks_reviews.data)
y = pdecks_reviews.target

# TODO: use k folds on the sparse matrix, rather than on raw data,
# if possible, b/c otherwise might inadvertently introduce bias
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.25, random_state=0)

X_train, X_test = np.copy(X), np.copy(X)
y_train, y_test = np.copy(y), np.copy(y)



#### EXTRACTING FEATURES #####

## TOKENIZATION ##
# create an instance of CountVectorize feature extractor
# using ngram_range flag, enable bigrams in addition to single words
count_vect = CountVectorizer(ngram_range=(1, 2))

# extract features from pdecks_reviews data
X_train_counts = count_vect.fit_transform(X_train)

## PART OF SPEECH TAGGING ##

## REMOVING PUNCTUATION ##

## REMOVING STOPWORDS ##

## STEMMING / LEMMATIZATION ##

## FREQUENCY DISTRIBUTIONS ##

## COLLOCATIONS, BIGRAMS, TRIGRAMS ##

## CHUNKING ##


## TF-IDF ##
# create an instance of TfidTransformer that performs both tf & idf
tfidf_transformer = TfidfTransformer()

# transform the pdecks_reviews features
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


## CLASSIFIER ##
# Linear SVC, recommended by sklearn machine learning map
# clf = Classifier().fit(features_matrix, targets_vector)
clf = LinearSVC().fit(X_train_tfidf, y_train)

new_doc = ['I love gluten-free foods. This restaurant is the best.']

X_new_counts = count_vect.transform(new_doc)  # transform only, as vectorizer is fit to training data
X_new_tfidf = tfidf_transformer.transform(X_new_counts)


# TEST: predict label (target) for new document
predicted = clf.predict(X_new_tfidf)

print
print "-- Test document --"
for doc, category in zip(new_doc, predicted):
  # retrieve label name
    print "%r => %s" % (doc, pdecks_reviews.target_names[category])


## VERIFY CLASSIFIER ACCURACY ON TRAINING DATA ##

count = 0
inaccurate = 0

predicted_array = np.array([])
for x_var, y_actual in zip(X_train, y_train):
  # print "###################"
  # print "x_var: \n", x_var
  # print
  X_new_counts = count_vect.transform([x_var])  # transform only, as
  X_new_tfidf = tfidf_transformer.transform(X_new_counts)
  # print X_new_tfidf.shape   # (1, 5646)
  predicted = clf.predict(X_new_tfidf)
  predicted_array = np.append(predicted_array, predicted)
  # print "predicted: %r, actual: %r" % (predicted, y_actual)
  # print "###################"
  if y_actual != predicted[0]:
    inaccurate += 1
  count += 1

print
print "-- Accuracy check by hand --"
print "PERCENT INACCURATE:", (inaccurate/(count*1.0))*100
print
print "-- Numpy mean calculation --"
print "PERCENT ACCURATE:", np.mean(predicted_array == y_test)*100


## CROSS-VALIDATING CLASSIFIERS ##
# randomly partition data set into 10 folds ignoring the classification variable
# b/c we want to see how the classifier performs in this real-world situation


# start with k=2, eventually increase to k=10 with larger dataset
avg_scores = {}
for k in range(2,6):
  avg_scores[k] = {}

for k in range(2, 6):
  n_fold = k
  # run k_fold 50 times at each value of k (2, 3, 4, 5)
  # take average score for each fold, keeping track of scores in dictionary
  k_dict = {}
  for i in range(1, n_fold+1):
    k_dict[i] = []

  for j in range(1, len(X)+1):
    k_fold = KFold(n=len(X), n_folds=n_fold, shuffle=True, random_state=random.randint(1,101))

    # k_fold_scores = [clf.fit(X_train_tfidf[train], y[train]).score(X_train_tfidf[test], y[test]) for train, test in k_fold]

    # print
    # print "-- Results of k-fold training and testing --"
    # print
    # print "Number of folds: {}".format(n_fold)
    # k_fold_scores = np.array([])


    i = 1
    for train, test in k_fold:
        score = clf.fit(X_train_tfidf[train], y[train]).score(X_train_tfidf[test], y[test])
        k_dict[i].append(score)
        # print "Fold: {} | Score:  {:.4f}".format(i, score)
        # k_fold_scores = np.append(k_fold_scores, score)
        i += 1
  # import pdb; pdb.set_trace()
  avg_scores[k] = k_dict

print
print '-- K-Fold Cross Validation --------'
print '-- Mean Scores for {} Iterations --'.format(j)
print
for k in range(2,6):
  print '-- k = {} --'.format(k)
  for i in range(1, k+1):
    print 'Fold: {} | Mean Score: {}'.format(i, np.array(avg_scores[k][i]).mean())
print
  
# print "Fold: {} | Score:  {:.4f}".format(i, score)


## CREATING PIPELINES FOR CLASSIFIERS ##
# Pipeline([(vectorizer), (transformer), (classifier)])
# text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
#                      ('tfidf', TfidfTransformer()),
#                      ('clf', LinearSVC()),
#                      ])

# # train the model
# text_clf = text_clf.fit(X_train, y_train)
# predicted = text_clf.predict(new_doc)

# for doc, category in zip(new_doc, predicted):
#     print "%r => %s" % (doc, pdecks_reviews.target_names[category])





# k_fold = KFold(n=len(X), n_folds=5, shuffle=True, random_state=random.randint(1,101))