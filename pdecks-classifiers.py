"""
Performing NLP using scikit-learn and NLTK.

by Patricia Decker, 11/2015, Hackbright Academy Independent Project

1. Classify Document Category
LinearSVC classifier that takes features vectors consisting of tokenized
reviews that have been converted to numerical values (counts) and
transformed to account for term frequency and inverse document frequency
(tf-idf). Tested on toy data set: 45 hand-labeled reviews that, for the
most part, already contain the word 'gluten'.

2. Perform Sentiment Analysis on Business review
Use NLTK on full-review text to target sentences related to category of
interest and assess sentiment of those target sentences. Generates a
sentiment score for the category based on a probability from 0.0 to 1.0,
where 1.0 is good and 0.0 is bad.
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
from sklearn.externals import joblib

# import reviewfilter as rf


#### LOAD DATA ###############################################################

# directory containing toy data set: reviews by pdecks as .txt files
# must be preprocessed with 'preprocess-reviews.py' if the .txt files
# contain more than just review information delimited on pipes
container_path = 'pdecks-reviews/'
categories = ['bad', 'excellent', 'good', 'limited', 'neutral', 'shady']

pickle_path_SVC = 'classifiers/LinearSVC/linearSVC.pkl'

# categories = ['gluten-free',
#               'paleo',
#               'pescatarian',
#               'vegan',
#               'vegetarian',
#               'omnivorous',
#               'kosher',
#               'halal']


def loads_pdecks_reviews(container_path, categories):
    # load the list of files matching the categories
    # BEWARE OF LURKING .DS_Store files!! those are not 'utf-8'
    # and will throw a UnicodeDecodeError

    documents = sk_base.load_files(container_path,
                                   categories=categories,
                                   encoding='utf-8')
    return documents


def define_train_test_sets(documents):
    """Takes complete dataset and splits it into training and testing set."""
    # TODO: update how data is split into a test set and a training set
    # Define the training dataset
    train_docs = documents
    test_docs = train_docs

    return (train_docs, test_docs)


def create_vectorizer(X_train):
    """Returns a sklearn vectorizer fit to training data.

    Input is a numpy array of training data."""
    # create an instance of CountVectorize feature extractor
    # using ngram_range flag, enable bigrams in addition to single words
    count_vect = CountVectorizer(ngram_range=(1, 2))
    # extract features from training documents' data
    X_train_counts = count_vect.fit_transform(X_train)

    return count_vect


def create_transformer(X_train_counts):
    """Returns a sklearn transformer fit to training data.

    Input is a numpy array of training data feature counts."""
    # create an instance of TfidTransformer that performs both tf & idf
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    return tfidf_transformer


## CREATE AND TRAIN DOCUMENT CLASSIFIER ##
def create_train_classifier(train_docs):
    """Takes documents (sklearn bunch) and returns a trained classifier
    and its vectorizer and transformer."""
    X = np.array(train_docs.data)
    y = train_docs.target


    # TODO: use k folds on the sparse matrix, rather than on raw data,
    # if possible, b/c otherwise might inadvertently introduce bias
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.25, random_state=0)
    X_train, X_test = np.copy(X), np.copy(X)
    y_train, y_test = np.copy(y), np.copy(y)

    ## EXTRACTING FEATURES ##
    # TOKENIZATION
    count_vect = create_vectorizer(X_train)
    X_train_counts = count_vect.transform(X_train)

    ## TF-IDF ##
    tfidf_transformer = create_transformer(X_train_counts)
    X_train_tfidf = tfidf_transformer.transform(X_train_counts)


    ## CLASSIFIER ##
    # Linear SVC, recommended by sklearn machine learning map
    # clf = Classifier().fit(features_matrix, targets_vector)
    clf = LinearSVC().fit(X_train_tfidf, y_train)

    ## CREATING PIPELINES FOR CLASSIFIERS #
    # Pipeline([(vectorizer), (transformer), (classifier)])
    pipeline_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                         ('tfidf', TfidfTransformer()),
                         ('clf', LinearSVC()),
                        ])
    # train the pipeline
    pipeline_clf = pipeline_clf.fit(X_train, y_train)

    return (count_vect, tfidf_transformer, clf, pipeline_clf)


## PERSIST THE MODEL ##
def persist_classifier(pipeline_clf, pickle_path):
    """Use joblib to pickle the pipeline model to disk."""
    joblib.dump(pipeline_clf, pickle_path)
    return


## REVIVE CLASSIFIER TO CATEGORIZE NEW REVIEWS ##
def revives_model(pickle_path):
    """Takes the name of the pickled object and returns the revived model.

    ex: clf_revive = pickle.loads(pdecks_trained_classifier)
    """
    model_clone = joblib.load(pickled_name)
    return model_clone


# TODO: Make sure this is garbage ...
# predicted = text_clf.predict(new_doc)

# for doc, category in zip(new_doc, predicted):
#     print "%r => %s" % (doc, pdecks_reviews.target_names[category])

# k_fold = KFold(n=len(X), n_folds=5, shuffle=True, random_state=random.randint(1,101))



## CLASSIFY NEW BUSINESS
def categorizes_review(review_text, count_vect, tfidf_transformer, clf):
    """Takes an array containing review text and returns the most relevant
    category for the review.

    new_doc_test = ['This restaurant has gluten-free foods.']
    new_doc_cv = count_vect.transform(new_doc_test)
    new_doc_tfidf = tfidf_transformer.transform(new_doc_cv)
    new_doc_category = clf_revive.predict(new_doc_tfidf)
    print "%s => %s" % (new_doc_test[0], categories[new_doc_category[0]])
    """

    # TODO: decide if it is necessary to continually pickle/unpickle every time
    # the classifier is used

    # TODO: unpickle classifier
    # clf_revive = revives_model(pdecks_trained_classifier)

    text_to_classify = review_text
    text_to_classify_counts = count_vect.transform(text_to_classify)
    text_to_classify_tfidf = tfidf_transformer.transform(text_to_classify_counts)
    new_doc_category = clf.predict(text_to_classify_tfidf)

    # TODO: pickle classifier
    # pdecks_trained_classifier = pickles_model(clf_revive)

    return new_doc_category


def get_category_name(category_id):
    """Takes a category index and returns a category name."""
    return categories[category_id]



#### SENTIMENT ANALYSIS CLASSIFIER ####

## TOKENIZATION ##

## PART OF SPEECH TAGGING ##

## REMOVING PUNCTUATION ##

## REMOVING STOPWORDS ##

## STEMMING / LEMMATIZATION ##

## FREQUENCY DISTRIBUTIONS ##

## COLLOCATIONS, BIGRAMS, TRIGRAMS ##

## CHUNKING ##




if __name__ == "__main__":

    # LOAD the training documents
    documents = loads_pdecks_reviews(container_path, categories)

    # DEFINE training and testing datasets
    train_docs, test_docs = define_train_test_sets(documents)
    X_train = np.array(train_docs.data)
    y_train = train_docs.target
    y_test = np.copy(y_train)

    # CREATE and TRAIN the classifier
    count_vect, tfidf_transformer, clf, pipeline_clf = create_train_classifier(documents)

    ## PERSIST THE MODEL ##
    persist_classifier(pipeline_clf, pickle_path_SVC)

    # TEST the classifier
    new_doc = ['I love gluten-free foods. This restaurant is the best.']
    new_doc_category_id = categorizes_review(new_doc,
                                             count_vect,
                                             tfidf_transformer,
                                             clf)

    new_doc_category = get_category_name(new_doc_category_id)
    new_doc_category_id_pipeline = pipeline_clf.predict(new_doc)
    new_doc_category_pipeline = get_category_name(new_doc_category_id_pipeline)

    print
    print "-- Test document --"
    print
    print "Using Vectorizer, Transformer, and Classifier:"
    # for doc, category in zip(new_doc, predicted):
    print "%r => %s" % (new_doc[0], new_doc_category)
    print
    print "Using Pipeline:"
    print "%r => %s" % (new_doc[0], new_doc_category_pipeline)


    ## VERIFY classifier accuracy on training data
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

    X = np.copy(X_train)
    y = np.copy(y_train)
    X_train_counts = count_vect.transform(X)
    X_train_tfidf = tfidf_transformer.transform(X_train_counts)

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
