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
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

# import reviewfilter as rf


#### LOAD DATA ###############################################################

# directory containing toy data set: reviews by pdecks as .txt files
# must be preprocessed with 'preprocess-reviews.py' if the .txt files
# contain more than just review information delimited on pipes

# toy data set: 45 reviews by author
# container_path = 'pdecks-reviews/'
# categories = ['bad', 'good', 'limited', 'shady', 'excellent']

# training data set from yelp academic database:
# 969 reviews containing the word 'gluten'
# 1000 reviews randomly sampled from 217,000 reviews NOT containing the word 'gluten'
container_path = './data/training/'
categories = ['gluten', 'unknown']

pickle_path_SVC = 'classifiers/LinearSVC/linearSVC.pkl'
pickle_path_v = 'classifiers/LSVCcomponents/vectorizer/linearSVCvectorizer.pkl'
pickle_path_t = 'classifiers/LSVCcomponents/transformer/linearSVCtransformer.pkl'
pickle_path_c = 'classifiers/LSVCcomponents/classifier/linearSVCclassifier.pkl'

# project goal (might be V2.0):
# categories = ['gluten-free',
#               'paleo',
#               'pescatarian',
#               'vegan',
#               'vegetarian',
#               'omnivorous',
#               'kosher',
#               'halal']

def loads_yelp_reviews(container_path, categories):
    """Load the training documents in data/training directory."""
    # TODO: update to handle pipes for keyword search directory .txt files
    # where format --> review_id | biz_id | biz_name | review_date | review_text
    documents = sk_base.load_files(container_path,
                                   categories=categories,
                                   encoding='utf-8')
    return documents


def loads_pdecks_reviews():
    """Load toy data set and check classifier working."""

    container_path = './pdecks-reviews/'
    categories = ['bad', 'excellent', 'good', 'limited', 'neutral', 'shady']

    documents = sk_base.load_files(container_path,
                                   categories=categories,
                                   encoding='utf-8')
    return documents


def bunch_to_np(documents):
    """
    Takes complete dataset and convert to np arrays.

    Documents input as a scikit bunch.
    """
    # TODO: update how data is split into a test set and a training set
    # Define the training dataset
    # train_docs = documents
    # test_docs = train_docs

    X = np.array(documents.data)
    y = documents.target

    return (X, y)


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
def create_train_classifier(X, y):
    """Takes documents (X) and targets (y), both np arrays, and returns a trained
    classifier and its vectorizer and transformer."""

    X_train = np.copy(X)
    y_train = np.copy(y)

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


## SCORE THE CLASSIFIER OVER K-Folds ##

def score_kfolds(X, y, num_folds=2, num_iter=1):
    """Perform cross-validation on sparse matrix (tf-idf).

    Returns a dictionary of the scores by fold."""

    count_vect = create_vectorizer(X)
    X_counts = count_vect.transform(X)

    tfidf_transformer = create_transformer(X_counts)
    X_tfidf = tfidf_transformer.transform(X_counts)

    clf = LinearSVC()

    print "Running score_kfolds with num_folds=%d, num_iter=%d" % (num_folds, num_iter)
    print "..."
    # randomly partition data set into 10 folds ignoring the classification variable
    # b/c we want to see how the classifier performs in this real-world situation

    # start with k=2, eventually increase to k=10 with larger dataset
    avg_scores = {}
    for k in range(2, num_folds + 1):
      avg_scores[k] = {}

    for k in range(2, num_folds + 1):
      n_fold = k
      print "Fold number %d ..." % k 
      # run k_fold num_iter number of times at each value of k (2, 3, ..., k)
      # take average score for each fold, keeping track of scores in dictionary
      k_dict = {}
      for i in range(1, n_fold + 1):
        k_dict[i] = []

      for j in range(1, num_iter +1):
        k_fold = KFold(n=len(X), n_folds=n_fold, shuffle=True, random_state=random.randint(1,101))
        # print "iteration: %d ..." % j
        i = 1
        for train, test in k_fold:
            score = clf.fit(X_tfidf[train], y[train]).score(X_tfidf[test], y[test])
            k_dict[i].append(score)
            # print "Fold: {} | Score:  {:.4f}".format(i, score)
            # k_fold_scores = np.append(k_fold_scores, score)
            i += 1
      # import pdb; pdb.set_trace()
      avg_scores[k] = k_dict

      print "Iterations for fold %d complete." % k

    print
    print '-- K-Fold Cross Validation --------'
    print '-- Mean Scores for {} Iterations --'.format(j)
    print
    for k in range(2, num_folds + 1):
      print '-- k = {} --'.format(k)
      for i in range(1, k+1):
        print 'Fold: {} | Mean Score: {}'.format(i, np.array(avg_scores[k][i]).mean())
    print

    return avg_scores


def tunes_parameters(X, y, n_fold=2):
    """Perform cross-validation on sparse matrix (tf-idf).

    Returns a dictionary of the scores by fold."""

    count_vect = create_vectorizer(X)
    X_counts = count_vect.transform(X)

    tfidf_transformer = create_transformer(X_counts)
    X_tfidf = tfidf_transformer.transform(X_counts)

    clf = LinearSVC()

    k_fold = KFold(n=len(X), n_folds=n_fold, shuffle=True, random_state=random.randint(1,101))

    # pass the entirity of the data, X_tfidf, to cross_val_score
    # cv is the number of folds for cross-validation
    # use classification accuracy as deciding metric
    scores = cross_val_score(clf, X_tfidf, y, cv=10, scoring='accuracy')
    print scores

    return scores


## PERSIST THE MODEL ##
def persist_pipeline(pipeline_clf, pickle_path):
    """Use joblib to pickle the pipeline model to disk."""
    joblib.dump(pipeline_clf, pickle_path)
    print 'Classifier pickled to directory: %s' % pickle_path
    print

    return


## REVIVE CLASSIFIER TO CATEGORIZE NEW REVIEWS ##
def revives_pipeline(pickle_name):
    """Takes the name of the pickled object and returns the revived model.

    ex: clf_revive = pickle.loads(pdecks_trained_classifier)
    """
    model_clone = joblib.load(pickle_name)
    return model_clone


## PERSIST A COMPONENT OF THE MODEL ##
def persist_component(component, pickle_path):
    """Use joblib to pickle the individual classifier components"""
    joblib.dump(component, pickle_path)
    print 'Component %s pickled to directory: %s' % (str(component), pickle_path)
    print
    return

## REVIVE COMPONENT ##
def revive_component(pickle_path):
    component_clone = joblib.load(pickle_path)
    return component_clone


## CLASSIFY NEW REVIEW
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
from nltk import word_tokenize, sent_tokenize

## STEMMING / LEMMATIZATION ##
from nltk.stem import WordNetLemmatizer

class LemmaTokenizer(object):
    """
    Stemming, lemmatizing, compound splitting, filtering based on POS, etc.
    are not included in sklearn but can be added by customizing either the
    tokenizer or the analyzer.

    Lemmatisation is closely related to stemming. The difference is that a
    stemmer operates on a single word without knowledge of the context, and
    therefore cannot discriminate between words which have different meanings
    depending on part of speech. However, stemmers are typically easier to
    implement and run faster, and the reduced accuracy may not matter for
    some applications.

    Decided not to use this in the end because Stanford NLP group suggests
    that there is no gain for doing such in English, which has weak
    morphology. Classification in other languages with strong morphology,
    such as German or Spanish, could benefit from a lemmatizer.

    Can be incorportated as follows:
    >>> vect = CountVectorizer(tokenizer=LemmaTokenizer())
    """
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


## PREPROCESSOR ##
class PennTreebankPunkt(object):
    """
    Create a custom preprocessor for use with sklearn's CountVectorizer using
    NLTK's Punkt Sentence Tokenizer (sent_tokenize) and NLTK's Penn Treebank
    Tokenizer (word_tokenize)

    This preprocessor aims to:
    1. Tokenize reviews on sentences with nltk.sent_tokenize
    2. Tokenize sentences with nltk.word_tokenize
    3. Correct contraction tokens (n't, 'll, etc.)
    4. Rejoin words into entire document delimited on white space --> string
    5. Optional: Store sentences in database with review_id info --> use with
       use_flag = "independent"
    """

    def __init__(self, use_flag="vectorizer"):
        self.pst = sent_tokenize()
        self.ptt = word_tokenize()
        self.use_flag = use_flag

    def __call__(self, doc):
        """
        if use_flag == vectorizer, return the entire document as a string
        else, if use_flag == 'independent', return the lists of sentences and
        the original words along with the preprocessed doc as a string.
        """
        # 1. tokenize into sentences
        sentence_list = self.pst(doc)

        # 2. tokenize into words
        word_list = []
        for sentence in sentence_list:
            word_list.extend(self.ptt(sentence))
        # word_list = [word_list.extend(self.ptt(sentence)) for sentence in sentence_list]

        # 2a. save original list of words
        original_word_list = word_list[:]

        # 3. correct contraction tokens, uses slice assignment
        # word_list[:] = [check_contraction(word) for word in word_list]
        # second form doesn't require the create of a temporary list and an
        # assignment of it to replace the original, although it does
        # require more indexing operations
        for i, word in enumerate(word_list):
            word_list[i] = check_contraction(word)

        # 4. rejoin words into single string
        prepocessed_doc = " ".join(word_list)

        if self.use_flag == 'independent':
            return (sentence_list, original_word_list, preprocessed_doc)
        else:
            return preprocessed_doc


    def check_contraction(word):
        """Converts contraction fragments to their equivalent words"""
        contraction_dict = {"'m": 'am',
                            "n't": 'not',
                            "'ll": 'will',
                            "ca":   'can',
                            "gon":  'going',
                            "na":   'to',
                            }
        if contraction_dict[word]:
            word = contraction_dict[word]
        return word


## TOKENIZATION ##
# Caution: when tokenizing a Unicode string, make sure you are not using an
# encoded version of the string (it may be necessary to decode it first,
# e.g. with s.decode("utf8").
def vectorize(X_docs, vocab=None):
    """Vectorizer for use with sentiment analysis.

    X_docs is a numpy array of documents to be vectorized.

    vocab is the vectorizer vocabulary, vectorizer.vocabulary_

    note on preprocessor:
    a callable that takes an entire document as input (as a single string),
    and returns a possibly transformed version of the document, still as an
    entire string."""

    vectorizer = CountVectorizer(strip_accents='unicode',
                                 stop_words='english',
                                 decode_error='strict',
                                 ngram_range=(1, 2),
                                 preprocessor=PennTreebankPunkt())
    if vocab:
        vectorizer = CountVectorizer(strip_accents='unicode',
                                     stop_words="english",
                                     decode_error='strict',
                                     ngram_range=(1, 2),
                                     preprocessor=PennTreebankPunkt(),
                                     vocabulary=vocab)

    X = vectorizer.fit_transform(X_docs)
    return vectorizer.vocabulary_, X


## FEATURE EXTRACTION ##
from sklearn.feature_selection import chi2

def sorted_features (cat_code, V, X, y, topN):
    """
    Use chi-square test scores to select top N features from vectorizer.

    Aims to simplify the classifier by training on only the most important
    features. The relative importance of the features is important in text
    classification. Chi-square feature selection can be used to rank features
    but is not appropriate for making statements about statistical dependence
    or independence of variables. [see Stanford NLP]

    cat_code: the 4-character category code (e.g., 'gltn', 'pleo')
    V: vectorizer vocabulary, vectorizer.vocabulary_
    X: numpy sparse matrix of vectorized documents
    y: numpy array of labels (target vector)

    Returns a list of the topN features.
    """
    # define the inverse dictionary for the vocabulary
    # key = rank, value = word
    iv = {v:k for k, v in V.items()}

    chi2_scores = chi2(X, y)[0]

    top_features = [(x[1], iv[x[0]], x[0])
                    for x in sorted(enumerate(chi2_scores),
                    key=operator.itemgetter(1), reverse=True)]

    print "TOP %s FEATURES FOR: %s" % (topN, cat_code)
    for top_feature in top_features[0:topN]:
        print "%7.3f %s (%d)" % (top_feature[0], top_feature[1], top_feature[2])

    return [x[1] for x in top_features]


## FREQUENCY DISTRIBUTIONS?? ##

## Helper function for checking if input string represents an int
def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False



if __name__ == "__main__":
    ## TRAIN AND PERSIST CLASSIFIER ##
    to_train = raw_input("Train the classifier? Y or N >> ")
    if to_train.lower() == 'y':
        # LOAD the training documents
        # documents = loads_pdecks_reviews(container_path, categories)
        documents = loads_yelp_reviews(container_path, categories)
        X, y = bunch_to_np(documents)

        # Train the model and cross-validate
        num_folds = raw_input("Enter a number of folds (2-10): ")
        num_iter = raw_input("Enter a number of iterations (1-50): ")
        print

        while not represents_int(num_folds) or not represents_int(num_iter):
            num_folds = raw_input("Enter a number of folds (2-10): ")
            num_iter = raw_input("Enter a number of iterations (1-50): ")
            print

        num_folds = int(num_folds)
        num_iter = int(num_iter)
        fold_avg_scores = score_kfolds(X, y, num_folds, num_iter)

        scores = tunes_parameters(X, y, 10)

        # CREATE and TRAIN the classifier
        X, y = bunch_to_np(documents)
        count_vect, tfidf_transformer, clf, pipeline_clf = create_train_classifier(X, y)



        ##TEST the classifier
        new_doc = ['I love gluten-free foods. This restaurant is the best.']
        # new_doc_category_id = categorizes_review(new_doc,
        #                                          count_vect,
        #                                          tfidf_transformer,
        #                                          clf)

        # new_doc_category = get_category_name(new_doc_category_id)
        new_doc_category_id_pipeline = pipeline_clf.predict(new_doc)
        new_doc_category_pipeline = get_category_name(new_doc_category_id_pipeline)

        print
        print "-- Test document --"
        print
        # print "Using Vectorizer, Transformer, and Classifier:"
        # for doc, category in zip(new_doc, predicted):
        # print "%r => %s" % (new_doc[0], new_doc_category)
        print
        print "Using Pipeline:"
        print "%r => %s" % (new_doc[0], new_doc_category_pipeline)


        ## PERSIST THE MODEL ##
        decision = raw_input("Would you like to persist the pipeline classifier? (Y) or (N) >>")
        if decision.lower() == 'y':
            persist_pipeline(pipeline_clf, pickle_path_SVC)
        else:
            print 'Classifier not pickled.'
            print

        ## PERSIST THE COMPONENTS ##
        decision = raw_input("Would you like to persist the vectorizer? (Y) or (N) >>")
        if decision.lower() == 'y':
            persist_component(count_vect, pickle_path_v)
        else:
            print 'Vectorizer not pickled.'

        decision = raw_input("Would you like to persist the transformer? (Y) or (N) >>")
        if decision.lower() == 'y':
            persist_component(tfidf_transformer, pickle_path_t)
        else:
            print 'Transformer not pickled.'

        decision = raw_input("Would you like to persist the classifier? (Y) or (N) >>")
        if decision.lower() == 'y':
            persist_component(clf, pickle_path_c)
        else:
            print 'Classifier not pickled.'


    ## CHECK PERFORMANCE ON TOY DATA SET ##
    else:
        to_test = raw_input("Check the pipeline classifier on the toy data set? Y or N >>")
        if to_test.lower() == 'y':
            documents_pd = loads_pdecks_reviews()
            X_pd, y_pd = bunch_to_np(documents_pd)
            # transform toy dataset labels to 0 = gluten, 1 = unknown
            y_trans_pd = np.array([])
            for target in y_pd:
                if target == 0 or target == 4:
                    # unknown
                    y_trans_pd = np.append(y_trans_pd, 1)
                else:
                    # gluten
                    y_trans_pd = np.append(y_trans_pd, 0)

            pipeline_clf = revives_pipeline(pickle_path_SVC)

            inaccurate = 0
            predicted_pd = []
            for i in range(0, len(X_pd)):
                predicted = pipeline_clf.predict([X_pd[i]])
                # print "predicted: %d, actual: %r" % (predicted, y_trans_pd[i])
                if y_trans_pd[i] != predicted:
                    inaccurate += 1
                i += 1
                predicted_pd.append(predicted)
            print "-- Accuracy check of toy dataset --"
            print "PERCENT INACCURATE: ", (inaccurate/(len(y_trans_pd)))*100
            for i in range(0, len(y_trans_pd)):
                print "Index i: %s" % i
                print "Predicted: %s" % predicted_pd[i][0]
                print "Actual: %s" % str(int(y_trans_pd[i]))
                print '-'*20


    # ## VERIFY classifier accuracy on training data
    # count = 0
    # inaccurate = 0

    # predicted_array = np.array([])
    # for x_var, y_actual in zip(X_train, y_train):
    #   # print "###################"
    #   # print "x_var: \n", x_var
    #   # print
    #   X_new_counts = count_vect.transform([x_var])  # transform only, as
    #   X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    #   # print X_new_tfidf.shape   # (1, 5646)
    #   predicted = clf.predict(X_new_tfidf)
    #   predicted_array = np.append(predicted_array, predicted)
    #   # print "predicted: %r, actual: %r" % (predicted, y_actual)
    #   # print "###################"
    #   if y_actual != predicted[0]:
    #     inaccurate += 1
    #   count += 1

    # print
    # print "-- Accuracy check by hand --"
    # print "PERCENT INACCURATE:", (inaccurate/(count*1.0))*100
    # print
    # print "-- Numpy mean calculation --"
    # print "PERCENT ACCURATE:", np.mean(predicted_array == y_test)*100

### MOVED INTO FUNCTION ABOVE score_kfolds ###
    # ## CROSS-VALIDATING CLASSIFIERS ##
    # # randomly partition data set into 10 folds ignoring the classification variable
    # # b/c we want to see how the classifier performs in this real-world situation

    # X = np.copy(X_train)
    # y = np.copy(y_train)
    # X_train_counts = count_vect.transform(X)
    # X_train_tfidf = tfidf_transformer.transform(X_train_counts)

    # # start with k=2, eventually increase to k=10 with larger dataset
    # avg_scores = {}
    # for k in range(2,6):
    #   avg_scores[k] = {}

    # for k in range(2, 6):
    #   n_fold = k
    #   # run k_fold 50 times at each value of k (2, 3, 4, 5)
    #   # take average score for each fold, keeping track of scores in dictionary
    #   k_dict = {}
    #   for i in range(1, n_fold+1):
    #     k_dict[i] = []

    #   for j in range(1, len(X)+1):
    #     k_fold = KFold(n=len(X), n_folds=n_fold, shuffle=True, random_state=random.randint(1,101))

    #     # k_fold_scores = [clf.fit(X_train_tfidf[train], y[train]).score(X_train_tfidf[test], y[test]) for train, test in k_fold]

    #     # print
    #     # print "-- Results of k-fold training and testing --"
    #     # print
    #     # print "Number of folds: {}".format(n_fold)
    #     # k_fold_scores = np.array([])


    #     i = 1
    #     for train, test in k_fold:
    #         score = clf.fit(X_train_tfidf[train], y[train]).score(X_train_tfidf[test], y[test])
    #         k_dict[i].append(score)
    #         # print "Fold: {} | Score:  {:.4f}".format(i, score)
    #         # k_fold_scores = np.append(k_fold_scores, score)
    #         i += 1
    #   # import pdb; pdb.set_trace()
    #   avg_scores[k] = k_dict

    # print
    # print '-- K-Fold Cross Validation --------'
    # print '-- Mean Scores for {} Iterations --'.format(j)
    # print
    # for k in range(2,6):
    #   print '-- k = {} --'.format(k)
    #   for i in range(1, k+1):
    #     print 'Fold: {} | Mean Score: {}'.format(i, np.array(avg_scores[k][i]).mean())
    # print

    # # print "Fold: {} | Score:  {:.4f}".format(i, score)
