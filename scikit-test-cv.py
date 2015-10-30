"""
Performing NLP using scikit-learn. Supervised machine learning.

by Patricia Decker, 10/28/2015, Hackbright Academy Independent Project

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
"""
import numpy as np
from sklearn.datasets import base as sk_base
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

import pdb

# import reviewfilter as rf


#### LOAD DATA ########################################################

# directory containing toy data set: reviews by pdecks as .txt files
# must be preprocessed with 'preprocess-reviews.py'
container_path = '/Users/pdecks/hackbright/project/Yelp/mvp/pdecks-reviews/'

categories = ['bad', 'excellent', 'good', 'limited', 'neutral', 'shady']

# load the list of files matching the categories
# BEWARE OF LURKING .DS_Store files!! those are not 'utf-8'
# and will throw a UnicodeDecodeError
pdecks_reviews = sk_base.load_files(container_path,
                                  categories=categories,
                                  encoding='utf-8')



## for un-processed .txt. ##
# # create list of filenames
# filelist = rf.generate_filelist(container_pay)

# # convert .txt review files to list of dictionaries (matches Yelp JSON)
# my_reviews = rf.generate_reviews_dict(filelist)

# for review in my_reviews:
#     print review['business_id']

## end un-processed .txt ##


#### EXTRACTING FEATURES #####

## TOKENIZATION ##
# create an instance of CountVectorize feature extractor
# using ngram_range flage, enable bigrams in addition to single words
count_vect = CountVectorizer(ngram_range=(1, 2))

# extract features from pdecks_reviews data
# X_train_counts = count_vect.fit_transform(X_train)
pdecks_counts = count_vect.fit_transform(pdecks_reviews.data)

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
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
pdecks_tfidf = tfidf_transformer.fit_transform(pdecks_counts)

# define cross-validation iterator
# cv = ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.2, random_state=0)

# Split the dataset into a test set and a training set
pdb.set_trace()  # debugging
X = pdecks_tfidf
y = pdecks_reviews.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

## DEVELOP CLASSIFIER
# Linear SVC, recommended by sklearn machine learning map
# clf = Classifier().fit(features_matrix, targets_vector)
clf = LinearSVC().fit(X_train, y_train)

new_doc = ['I love gluten-free foods. This restaurant is the best.']

X_new_counts = count_vect.transform(new_doc)  # transform only, as vectorizer is fit to training data
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

# predict label (target) for new document
predicted = clf.predict(X_new_tfidf)

# retrieve label name
for doc, category in zip(new_doc, predicted):
    print "%r => %s" % (doc, pdecks_reviews.target_names[category])

## CROSS-VALIDATING CLASSIFIERS ##


## CREATING PIPELINES FOR CLASSIFIERS ##
# Pipeline([(vectorizer), (transformer), (classifier)])
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                     ('tfidf', TfidfTransformer()),
                     ('clf', LinearSVC()),
                     ])

# train the model
text_clf = text_clf.fit(X_train, y_train)
predicted = text_clf.predict(new_doc)

for doc, category in zip(new_doc, predicted):
    print "%r => %s" % (doc, pdecks_reviews.target_names[category])


## EVALUATE PERFORMANCE ##
predicted = text_clf.predict(X_test)
np.mean(predicted == y_test) # was 33.3% with 75-25 split on 45 reviews. Ouch.

## TUNE THE HYPERPARAMETERS ##
# apply the cross-validation iterator on the training set
tuned_parameters = [{'C': [1, 10, 100, 1000],
                     'penalty': ['l1', 'l2'],
                     'tol': [1e-3, 1e-4, 1e-5]
                     }]

scores = ['precision', 'recall']

for score in scores:
    print "# Tuning hyper-parameters for %s" % score
    print

    clf = GridSearchCV(LinearSVC(), tuned_parameters, cv=10,
                       scoring='%s_weighted' % score)

    clf.fit(X_train, y_train)

    print "Best parameters set found on development set:"
    print
    print clf.best_params_
    print
    print "Grid scores on development set:"
    print
    for params, mean_score, scores in clf.grid_scores_:
        print "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params)

    print

    print "Detailed classification report:"
    print
    print "The model is trained on the full development set."
    print "The scores are computed on the full evaluatin set."
    print
    y_true, y_pred = y_test, clf.predict(X_test)
    print classification_report(y_true, y_pred)
    print
    