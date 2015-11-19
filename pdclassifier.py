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
import math
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import gridspec

from sklearn.datasets import base as sk_base
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

from nltk import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer


# import reviewfilter as rf

## DIRECTORIES FOR PICKLING CLASSIFIER & COMPONENTS ##########################

pickle_path_SVC = 'classifiers/LinearSVC/linearSVC.pkl'
pickle_path_v = 'classifiers/LSVCcomponents/vectorizer/linearSVCvectorizer.pkl'
pickle_path_t = 'classifiers/LSVCcomponents/transformer/linearSVCtransformer.pkl'
pickle_path_c = 'classifiers/LSVCcomponents/classifier/linearSVCclassifier.pkl'

pickle_path_rfc = 'classifiers/random_forest/classifier/randomforest.pkl'
pickle_path_rfv = 'classifiers/random_forest/vectorizer/randomforest.pkl'

pickle_path_SA_v = './classifiers/SentimentComponents/gltn_vectorizer/vectorizer.pkl'
pickle_path_SA_gltn = './classifiers/SentimentComponents/gltn_classifier/gltn_classifier.pkl'
pickle_path_SANB_gltn = './classifiers/SentimentComponents/gltn_naivebayes/gltn_naivebayes.pkl'

#### LOAD DATA ###############################################################

# directory containing toy data set: reviews by pdecks as .txt files
# must be preprocessed with 'preprocess-reviews.py' if the .txt files
# contain more than just review information delimited on pipes

# toy data set: 45 reviews by author
container_path = 'pdecks-reviews/'

# training data set from yelp academic database:
# 969 reviews containing the word 'gluten'
# 1000 reviews randomly sampled from 217,000 reviews NOT containing the word 'gluten'


container_path_pd = './pdecks-reviews/'
categories_pd = ['bad', 'excellent', 'good', 'limited', 'neutral', 'shady']


def loads_yelp_reviews(container_path, categories=None):
    # categories = ['gluten', 'unknown']
    # load all categories for random forest
    if not categories:
        categories = ['unknown', 'gluten', 'allergy', 'paleo', 'kosher', 'vegan']

    """Load the training documents in container_path directory."""
    # TODO: update to handle pipes for keyword search directory .txt files
    # where format --> review_id | biz_id | biz_name | review_date | review_text
    documents = sk_base.load_files(container_path,
                                   categories=categories,
                                   encoding='utf-8')
    return documents


def loads_pdecks_reviews(container_path=container_path_pd, categories=categories_pd):
    """Load toy data set and check classifier working."""

    documents = sk_base.load_files(container_path,
                                   categories=categories,
                                   encoding='utf-8')
    return documents


def bunch_to_np(documents, class_type=None):
    """
    Takes complete dataset and convert to np arrays.

    Documents input as a scikit bunch.

    For class_type="sentiment", the text files loaded into documents.data
    should contain the yelp stars as the first pipe-delimited value. Split
    this score off the text and store as the target (y), and correct data
    to only contain the review text, still returning (X, y)
    """

    if class_type == 'sentiment':
        X = []
        y = []
        for i, val in enumerate(documents.data):
            split_text = val.split("|")
            y.append(int(split_text[0][-1]))
            X.append(split_text[5])
        X = np.array(X)
    else:
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
# add number of features ...

def score_kfolds(X, y, min_num_folds=2, max_num_folds=2, num_iter=1, atype=None, num_feats=None):
    """Perform cross-validation on sparse matrix (tf-idf).

    Returns a dictionary of the scores by fold.
    atype: if "sentiment", cross-validate sentiment analysis model
           which assumes the input X is already transformed into a
           sparse matrix of tf-idf values. if None, assumes X needs
           to first be vectorized.

    """
    if atype is None:
        count_vect = create_vectorizer(X)
        X_counts = count_vect.transform(X)

        tfidf_transformer = create_transformer(X_counts)
        X_tfidf = tfidf_transformer.transform(X_counts)
    else:
        X_tfidf = X

    clf = LinearSVC()

    if num_feats:
        print "Number of features:", num_feats
        print
    print "Running score_kfolds with min_num_folds=%d, max_num_folds=%d, num_iter=%d" % (min_num_folds, max_num_folds, num_iter)
    print "..."
    # randomly partition data set into 10 folds ignoring the classification variable
    # b/c we want to see how the classifier performs in this real-world situation

    # start with k=2, eventually increase to k=10 with larger dataset
    avg_scores = {}
    all_avg_scores = {}
    for k in range(min_num_folds, max_num_folds + 1):
        avg_scores[k] = {}
        all_avg_scores[k] = {}
    for k in range(min_num_folds, max_num_folds + 1):
        n_fold = k
        print "Fold number %d ..." % k
        # run k_fold num_iter number of times at each value of k (2, 3, ..., k)
        # take average score for each fold, keeping track of scores in dictionary
        k_dict = {}
        all_scores = {}
        for i in range(1, n_fold + 1):
            k_dict[i] = []
            all_scores[i] = []

        for j in range(1, num_iter +1):
            k_fold = KFold(n=X_tfidf.shape[0], n_folds=n_fold, shuffle=True, random_state=random.randint(1,101))
            # print "iteration: %d ..." % j
            i = 1
            for train, test in k_fold:
                score = clf.fit(X_tfidf[train], y[train]).score(X_tfidf[test], y[test])
                y_predict = clf.predict(X_tfidf[test])
                accuracy = accuracy_score(y[test], y_predict)
                precision = precision_score(y[test], y_predict)
                recall = recall_score(y[test], y_predict)
                all_scores[i].append((accuracy, precision, recall))
                k_dict[i].append(score)
                # print "Fold: {} | Score:  {:.4f}".format(i, score)
                # k_fold_scores = np.append(k_fold_scores, score)
                i += 1

        avg_scores[k] = k_dict
        all_avg_scores[k] = all_scores

        print "Iterations for fold %d complete." % k

    print
    print '-- K-Fold Cross Validation --------'
    print '-- Mean Scores for {} Iterations --'.format(j)
    print
    for k in range(min_num_folds, max_num_folds + 1):
      print '-- k = {} --'.format(k)
      for i in range(1, k+1):
        print 'Fold: {} | Mean Score: {}'.format(i, np.array(avg_scores[k][i]).mean())
        if num_iter > 0:
            print 'Fold: {} | Mean Accuracy Score: {}'.format(i, np.mean(np.matrix(all_avg_scores[k][i])[:, 0].A1))
            print 'Fold: {} | Mean Precision Score: {}'.format(i, np.mean(np.matrix(all_avg_scores[k][i])[:, 1].A1))
            print 'Fold: {} | Mean Recall Score: {}'.format(i, np.mean(np.matrix(all_avg_scores[k][i])[:, 2].A1))
        else:
            pass
            # print 'Fold: {} | Mean Accuracy Score: {}'.format(i, all_avg_scores[k][i][num_iter - 1][0])
            # print 'Fold: {} | Mean Precision Score: {}'.format(i, all_avg_scores[k][i][num_iter - 1][1])
            # print 'Fold: {} | Mean Recall Score: {}'.format(i, all_avg_scores[k][i][num_iter - 1][2])
    print

    return (avg_scores, all_avg_scores)


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


## PERSIST A COMPONENT OF THE MODEL ##
def to_persist(items_to_pickle=None, pickling_paths=None):
    """
    Takes a list of components to pickle and a list of paths for each item
    to be pickled.
    """
    # todo: check pipeline case...
    if items_to_pickle and pickling_paths and len(items_to_pickle) == len(pickling_paths):
        for item, path in zip(items_to_pickle, pickling_paths):

            decision = raw_input("Would you like to persist %s?\nPath: %s\n(Y) or (N) >>"  % (str(item), str(path)))
            if decision.lower() == 'y':
                persist_component(item, path)
            else:
                print '%s not pickled.' % (str(item))
                print

    print "Persistance complete."
    return


def persist_component(component, pickle_path):
    """Use joblib to pickle the individual classifier components"""
    joblib.dump(component, pickle_path)
    print 'Component %s pickled to directory: %s' % (str(component), pickle_path)
    print
    return


## REVIVE COMPONENT ##
def revives_component(pickle_path):
    """Takes the name of the pickled object and returns the revived model.

    ex: clf_revive = pickle.loads(pdecks_trained_classifier)
    """
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

## STEMMING / LEMMATIZATION ##
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
        self.pst = sent_tokenize
        self.ptt = word_tokenize
        self.use_flag = use_flag

    def __call__(self, doc):
        """
        if use_flag == vectorizer, return the entire document as a string
        else, if use_flag == 'independent', return the lists of sentences and
        the original words along with the preprocessed doc as a string.

        >>> preprocessor = PennTreebankPunkt('word2vec')
        >>> text = "This is one sentence. Here is a second sentence."
        >>> sentence_list = preprocessor(text)
        >>> sentence_list
        [['This', 'is', 'one', 'sentence', '.'], ['Here', 'is', 'a', 'second', 'sentence', '.']]

        >>> text = "I'd rather not go, but I can't stay home. Won't you go with us?"
        >>> sentence_list = preprocessor(text)
        >>> sentence_list
        [['I', 'had', 'rather', 'not', 'go', ',', 'but', 'I', 'can', 'not', 'stay', 'home', '.'], ['Will', 'not', 'you', 'go', 'with', 'us', '?']]
        """
        # 1. tokenize into sentences
        raw_sentence_list = self.pst(doc)

        # 2. tokenize into words
        word_list = []

        for sentence in raw_sentence_list:
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
            word_list[i] = self.check_contraction(word)

        # 4. rejoin words into single string
        preprocessed_doc = " ".join(word_list)

        if self.use_flag == 'word2vec':

            sentence_list = []
            for sentence in raw_sentence_list:
                # tokenize into words
                words = self.ptt(sentence)
                # cleanup contractions
                for i, word in enumerate(words):
                    words[i] = self.check_contraction(word)
                # append list of words to list of sentences
                sentence_list.append(words)        
            return sentence_list
        elif self.use_flag == 'sentences':
            return raw_sentence_list
        else:
            return preprocessed_doc


    def check_contraction(self, word):
        """Converts contraction fragments to their equivalent words"""
        contraction_dict = {"'m": 'am',
                            "n't": 'not',
                            "'ll": 'will',
                            "ca": 'can',
                            "Ca": 'Can',
                            "Gon": 'Going',
                            "gon": 'going',
                            "na": 'to',
                            "'re": 'are',
                            "'ve": 'have',
                            "'d": 'had',
                            "wo": 'will',
                            "Wo": 'Will'
                            }

        if contraction_dict.get(word):
            word = contraction_dict[word]
        return word


## TOKENIZATION ##
# Caution: when tokenizing a Unicode string, make sure you are not using an
# encoded version of the string (it may be necessary to decode it first,
# e.g. with s.decode("utf8").

def vectorize(X_docs, vocab=None):
    """Vectorizer for use with random forests / sentiment analysis.

    X_docs is a numpy array of documents to be vectorized.

    vocab is the vectorizer vocabulary, vectorizer.vocabulary_

    The bytecode string is NOT in the vocabulary:
    byte_code = '\ufeff'
    byte_code in vect.vocabulary_.keys() --> False

    note on preprocessor:
    a callable that takes an entire document as input (as a single string),
    and returns a possibly transformed version of the document, still as an
    entire string."""

    vectorizer = TfidfVectorizer(strip_accents='unicode',
                                 stop_words='english',
                                 encoding='utf-8',
                                 decode_error='strict',
                                 ngram_range=(1, 1),
                                 preprocessor=PennTreebankPunkt())
    if vocab is not None:
        vectorizer = TfidfVectorizer(strip_accents='unicode',
                                     stop_words="english",
                                     encoding='utf-8',
                                     decode_error='strict',
                                     ngram_range=(1, 1),
                                     preprocessor=PennTreebankPunkt(),
                                     vocabulary=vocab)

    X = vectorizer.fit_transform(X_docs)
    return vectorizer, vectorizer.get_feature_names(), X


## FEATURE EXTRACTION ##

def sorted_features (feature_names, X_numerical, y, kBest=None):
    """
    Use chi-square test scores to select top N features from vectorizer.

    Aims to simplify the classifier by training on only the most important
    features. The relative importance of the features is important in text
    classification. Chi-square feature selection can be used to rank features
    but is not appropriate for making statements about statistical dependence
    or independence of variables. [see Stanford NLP]

    feature_names: vectorizer vocabulary, vectorizer.get_feature_names()
    X: numpy sparse matrix of vectorized documents (can also be tf-idf transformed)
    y: numpy array of labels (target vector)
    kBest: integer value of number of best features to extract

    Returns a list of the features as the words themselves in descending order
    of importance.
    """
    if not kBest:
        kBest = X_numerical.shape[1]
    ch2 = SelectKBest(chi2, kBest)

    X_numerical = ch2.fit_transform(X_numerical, y)

    # ch2.get_support() is an array of booleans, where True indicates that
    # the feature is among the bestK features
    # ch2.get_support(indicies=True) returns an array of the best feature indices
    # feature_names[i] maps the index to the vocabulary from the vectorizer to
    # retrieve the word at that index
    # best_feature_names is not ranked from best to worst
    best_feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
    best_feature_names = np.asarray(best_feature_names)

    # sort on score in descending order, but provide index and score.
    top_ranked_features = sorted(enumerate(ch2.scores_),key=lambda x:x[1], reverse=True)[:kBest]

    # zip(*top_ranked_features) splits the list of kBest (rank, score) tuples into 2 tuples:
    # 0: kBest-long tuple (best index, ... , least best index)
    # 1: kBest-long tuple (best score, ... , least best score)
    # top_ranked_features_indices = map(list,zip(*top_ranked_features))[0]
    top_ranked_features_indices = [x for x in zip(*top_ranked_features)[0]]

    # ranked from best to worst
    top_ranked_feature_names = np.asarray([feature_names[i] for i in top_ranked_features_indices])

    # P-values
    # for feature_pvalue in zip(np.asarray(train_vectorizer.get_feature_names())[top_ranked_features_indices],ch2.pvalues_[top_ranked_features_indices]):
    #     print feature_pvalue
    # # np.asarray(vectorizer.get_feature_names())[ch2.get_support()]

    return top_ranked_feature_names


def train_sentiment_vectorizer(dataset='pdecks'):
    """
    Run selected dataset through sentiment analysis vectorizer.

    Cross validates model for k folds and n features.

    Use with plot_sentiment_model_scores to select proper number of features

    Assumes results will be used with LinearSVC or MultinomialNB classifiers,
    not random forests, which has its own built-in feature importance ranker.
    """
    if dataset == 'yelp':
        documents = loads_yelp_reviews(container_path, categories=['gluten', 'unknown'])
        X, y = bunch_to_np(documents)
    elif dataset == 'stars':
        documents = loads_yelp_reviews(container_path="./data/sentiment", categories=['gluten'])
        X, y = bunch_to_np(documents, class_type='sentiment')
        X = X.tolist()

        # correct the data field and store additional data on documents
        true_index = 0
        while true_index < len(y):
            print "True Index: %d, y = %d, len(y): %s, len(X): %s" % (true_index, y[true_index], len(y), len(X))
            if y[true_index] == 5:
                y[true_index] = 1
                true_index += 1
            elif y[true_index] == 1:
                y[true_index] = 0
                true_index += 1
            else:
                y.pop(true_index)
                X.pop(true_index)
        X = np.array(X)
        y = np.array(y)
    else:
        documents = loads_pdecks_reviews()
        X, y = bunch_to_np(documents)

    # for cat in categories:
    # for cat in categories_pd:
    #     # documents = loads_yelp_reviews(container_path, [cat])

    # tranform y to a binary array (0 or 1 only where 1="good" and 0="bad")
    if dataset == 'pdecks':
        for i in range(y.shape[0]):
            if y[i] in [0, 3, 4, 5]:
                y[i] = 0
            else:
                y[i] = 1
    # if dataset == 'stars':
    #     for i in range(y.shape[0]):
    #         if y[i] == 5:
    #             y[i] = 1
    #         elif y[i] == 1:
    #             y[i] = 0
    #         else:
    #             np.delete(y, i)
    #             X.pop(i)
    #             # remove datapoints
    #         # if y[i] in [1, 2, 3]:
    #         #     y[i] = 0
    #         # else:
    #             # y[i] = 1


    vectorizer, feature_names, X_tfidf = vectorize(X)
    print "there are %d features" % len(feature_names)
    print

    min_num_folds, max_num_folds, num_iter = get_folds_and_iter()
    fold_avg_scores = score_kfolds(X_tfidf, y, min_num_folds, max_num_folds, num_iter, atype='sentiment')

    # Extract best features using chi2 test
    bestK = int(math.floor(len(feature_names) / 2))
    sorted_feats = sorted_features(feature_names, X_tfidf, y, kBest=bestK)

    print "Top %d Features from Chi-square Test for Category 'gltn':" % 10
    for feature in sorted_feats[0:10]:
        print "feature: ", feature
    print len(sorted_feats)

    user_choice = raw_input("Would you like to evaluate the best features? Y or N >> ")
    while user_choice.lower() not in ['y', 'n']:
        user_choice = raw_input("Would you like to evaluate the best features? Y or N >> ")
    if user_choice.lower() == 'y':
        avg_score_nfeats = {}
        scores_by_nfeats = {}
        if dataset == 'stars':
            feats_list = [100, 200, 300, 400, 500, 1000, 2500, 4500]
        elif dataset != 'yelp':
            feats_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        elif len(feature_names) > 25000:
            feats_list = [100, 200, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000]
        else:
            feats_list = [len(feature_names) / 10, len(feature_names) / 5, len(feature_names) / 2, len(feature_names)]

        # Check other features and update vocabulary
        for nfeats in feats_list:
            k_vectorizer, feature_names, X_tfidf = vectorize(X, sorted_feats[0:nfeats])
            print
            print "-"*20
            print "Perforing Cross-Validation with %d Features" % nfeats
            print "-"*20
            avg_score_nfeats[nfeats], scores_by_nfeats[nfeats] = score_kfolds(X_tfidf, y, min_num_folds, max_num_folds, num_iter, 'sentiment', nfeats)
        print
        print "Test successful"

    if user_choice.lower() == 'n':
        nfeats = ''
        while not represents_int(nfeats) or int(nfeats) > len(feature_names):
            nfeats = raw_input("Enter the number of features to use for the vectorizer: 1-%r" % str(len(feature_names)))
        nfeats = int(nfeats)
        vectorizer, feature_names, X_tfidf = vectorize(X, sorted_feats[0:nfeats])
        avg_score_nfeats, scores_by_nfeats = score_kfolds(X_tfidf, y, min_num_folds, max_num_folds, num_iter, 'sentiment', nfeats)

    return (vectorizer, scores_by_nfeats)


def train_sentiment_classifier():
    documents = loads_yelp_reviews(container_path="./data/sentiment", categories=['gluten'])
    X, y = bunch_to_np(documents, class_type='sentiment')
    X = X.tolist()

    # correct the data field and store additional data on documents
    true_index = 0
    while true_index < len(y):
        print "True Index: %d, y = %d, len(y): %s, len(X): %s" % (true_index, y[true_index], len(y), len(X))
        if y[true_index] == 5:
            y[true_index] = 1
            true_index += 1
        elif y[true_index] == 1:
            y[true_index] = 0
            true_index += 1
        else:
            y.pop(true_index)
            X.pop(true_index)
    X = np.array(X)
    y = np.array(y)
    vectorizer = revives_component(pickle_path_SA_v)
    # for cat in categories:
    # for cat in categories_pd:
    #     # documents = loads_yelp_reviews(container_path, [cat])
    X_tfidf = vectorizer.transform(X)

    """fit-transform the vectorized data on classifier"""
    ## CLASSIFIER ##
    # MultinomialNB, because ant to use predict_probab
    # clf = Classifier().fit(features_matrix, targets_vector)
    # SA_clf = LinearSVC().fit(X_tfidf, y)
    SA_clf = MultinomialNB().fit(X_tfidf, y)

    # TEST the classifier
    # import pdb; pdb.set_trace()
    new_doc = ['I love gluten-free foods. This restaurant is the best.']

    new_doc_tfidf = vectorizer.transform(new_doc)
    new_doc_predict = SA_clf.predict(new_doc_tfidf).tolist()
    predict = new_doc_predict[0]
    new_doc_proba = SA_clf.predict_proba(new_doc_tfidf).tolist()
    proba = new_doc_proba[0][predict]
    if predict == 1:
        print "the text was classified for 'gltn' as 'good' (1) with probability %f" % proba
    else:
        print "the text was classified for 'gltn' as 'bad' (0) with probability %f" % proba

    # PERSIST THE MODEL / COMPONENTS
    items_to_pickle = [SA_clf]
    pickling_paths = [pickle_path_SANB_gltn]
    # pickling_paths = [pickle_path_SA_gltn]
    to_persist(items_to_pickle=items_to_pickle, pickling_paths=pickling_paths)

    return


def predict_sentiment(text, categories=None, revive=True):
    """For a text, perform 'sentiment analysis' and return
    an array of predictions.

    >>> documents = loads_pdecks_reviews()
    >>> X = np.array(documents.data)
    >>> predictions = [predict_sentiment([doc]) for doc in X]
    >>> predictions[0:2]
    [[('gltn', 1, 0.5657259340602369)], [('gltn', 1, 0.6276715390190348)]]
    >>> X_list = X.tolist()
    >>> pairs = zip(X_list, predictions)
    (u"Even though people rave about the GF baked goods here, I am not such a fan because they use lots of soy, which I can't eat, either. I have enjoyed their coconut macaroons, but usually I just keep walking to Le Panier to get some French macarons instead.", [('gltn', 1, 0.5657259340602369)])

    This shows that even though the restaurant was categorized as 'good' (1),
    the probability that it is good is only 0.56, which is almost neutral.
    """
    if not isinstance(text, (np.ndarray, np.generic) ):
        if isinstance(text, list):
            text = np.array(text)
        else:
            text = np.array([text])
    if categories is None:
        categories = ['gltn']
    # TODO: keep pickle_paths in list
    prediction_list = []
    if revive == True:
        vectorizer = revives_component(pickle_path_SA_v)
        text_tfidf = vectorizer.transform(text)

        for category in categories:
            # revive correct classifier
            if category  == 'gltn':
                SA_clf = revives_component(pickle_path_SANB_gltn)
            elif category  == 'vgan':
                SA_clf = revives_component(pickle_path_SANB_vgan)
            elif category  == 'kshr':
                SA_clf = revives_component(pickle_path_SANB_kshr)
            elif category  == 'algy':
                SA_clf = revives_component(pickle_path_SANB_algy)
            elif category  == 'pleo':
                SA_clf = revives_component(pickle_path_SANB_pleo)
            else:
                pass
            prediction = SA_clf.predict(text_tfidf).tolist()
            pred_score = SA_clf.predict_proba(text_tfidf).tolist()
            # pred_score = SA_clf.decision_function(text_tfidf).tolist()
            prediction = int(prediction[0])
            # always take the 1st item (not 0th) because we want
            # to score probability with respect to 'good'
            pred_score = float(pred_score[0][1])
            # print "this is prediction %s and its type %r" % (prediction, type(prediction))
            prediction_list.append((category, prediction, pred_score))
    return prediction_list


def plot_sentiment_model_scores(scores_by_nfeats):
    """
    take the dictionary of scores returned by score_kfolds and generate plots

    plot nfreatures (independent) vs. precision, vs. accuracy, vs. recall
    """
    nfeats = scores_by_nfeats.keys()

    # retrieve number of folds
    min_num_folds = min(scores_by_nfeats[nfeats[0]].keys())
    max_num_folds = max(scores_by_nfeats[nfeats[0]].keys())
    num_iter = len(scores_by_nfeats[nfeats[0]][min_num_folds])

    mean_scores_by_kfolds = {}
    for k in range(min_num_folds, max_num_folds + 1):
        mean_scores_by_kfolds[k] = {'accuracy': {},
                                    'precision': {},
                                    'recall': {}
                                    }
    # calculate mean scores for fold j in max k folds
    for nfeat in nfeats:
        for k in range(min_num_folds, max_num_folds + 1):
            k_average = []
            for j in range (1, k + 1):
                # matrix: num_iter rows x 3 columns, where cols = accuracy, precision, recall
                # current_matrix = np.matrix(scores_by_nfeats[nfeats][k][j])
                # current_accuracy = np.mean(current_matrix[:, 0].A1)
                # current_precision = np.mean(current_matrix[:, 1].A1)
                # current_recall = np.mean(current_matrix[:, 2].A1)
                mean_scores = tuple([np.mean(np.array(x)) for x in zip(*scores_by_nfeats[nfeat][k][j])])
                k_average.append(mean_scores)

            k_average = [np.mean(np.array(x)) for x in zip(*k_average)]

            mean_scores_by_kfolds[k]['accuracy'][nfeat] = k_average[0]
            mean_scores_by_kfolds[k]['precision'][nfeat] = k_average[1]
            mean_scores_by_kfolds[k]['recall'][nfeat] = k_average[2]

    # reference an Axes object to keep drawing on the same subplot
    fig = plt.figure()
    ax_all_k = fig.add_subplot(111)

    # plotting styles (letter = marker type, dashes and dots are line type)
    pstyle = ['o-', 's-', 'v-', '*-', '+-', 'o--', 's--', 'v--', '*--', '+--', 'o-.', 's-.', 'v-.', '*-.', '+-.']

    for k in range(min_num_folds, max_num_folds + 1):

        d = mean_scores_by_kfolds[k]
        with plt.style.context('fivethirtyeight'):
            for data_name, data_dict in sorted(d.items(), key=lambda x: x[0]):
                data_points = zip(*sorted(data_dict.items()))
                label_str = "K=%d" % k
                labels = data_name + ' ' + label_str
                ax_all_k.plot(data_points[0], data_points[1], pstyle[k - min_num_folds], label=labels, linewidth=1)

    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,0.5,1))
    plt.xlabel("Number of Features (words)")
    title_str = "Accuracy, Precision, and Recall VS nfeatures\n K folds: %d-%d" % (min_num_folds, max_num_folds)
    plt.title(title_str)
    plt.legend(loc='lower left', fontsize='x-small')
    plt.show()

    # reference an Axes object to keep drawing on the same subplot
    num_subplots = max_num_folds - min_num_folds + 1
    if num_subplots > 1:
        fig, axs = plt.subplots(num_subplots, 1)
        i = 0
        for k in range(min_num_folds, max_num_folds + 1):
            ax = axs[i]
            d = mean_scores_by_kfolds[k]
            with plt.style.context('fivethirtyeight'):
                for data_name, data_dict in sorted(d.items(), key=lambda x: x[0]):
                    data_points = zip(*sorted(data_dict.items()))
                    label_str = "K=%d" % k
                    labels = data_name + ' ' + label_str
                    ax.plot(data_points[0], data_points[1], pstyle[k - min_num_folds], label=labels, linewidth=1)
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-small')
            # ax.legend(loc='lower right', fontsize='x-small')
            i += 1

        fig.suptitle(title_str, fontsize=10)
        plt.xlabel("Number of Features (words)")
        plt.show()

    return mean_scores_by_kfolds


## UNSUPERVISED LEARNING ##
# First, to train Word2Vec it is better not to remove stop words because
# the algorithm relies on the broader context of the sentence in order
# to produce high-quality word vectors.
from gensim.models import word2vec # for making model
from gensim.models import Word2Vec # for loading model
import logging
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
import time

def clustering_study():
    """using distributed word vectors created by the Word2Vec algorithm,
    a neural network implementation published by Google in 2013.

    Word2vec learns quickly relative to other models.
    Word2Vec does not need labels in order to create meaningful representations.

    >>> model.doesnt_match("man woman child kitchen".split())
    'kitchen'

    >>> model.doesnt_match("gluten vegan celiac dairy".split())
    'gluten'

    >>> model.doesnt_match("gluten celiac free wheat".split())
    'free'

    >>> model.most_similar("gluten")
    [(u'restaurants', 0.966518759727478), (u'except', 0.9627533555030823), (u'brunch', 0.9542723298072815), (u'their', 0.952192485332489), (u'seems', 0.9517408013343811), (u'prefer', 0.9516568183898926), (u'Indian', 0.9506375789642334), (u'Now', 0.9481799602508545), (u'ambiance', 0.9477816820144653), (u'high', 0.9476841688156128)]

    >>> model.most_similar("vegan")
    [(u'burgers', 0.8741724491119385), (u'Their', 0.864501953125), (u'vegetarian', 0.8616198301315308), (u'best', 0.8526089191436768), (u'tastes', 0.8514992594718933), (u'pizza', 0.8467479944229126), (u'etc', 0.8420628309249878), (u'interesting', 0.8413756489753723), (u'filling', 0.8396750688552856), (u'They', 0.8361315727233887)]

    """
    container_path = './data/random_forest/'
    print "... Loading data from %s ..." % container_path

    documents = loads_yelp_reviews(container_path)
    y = documents.target
    for i in range(len(y)):
        if y[i] in [0, 3, 4, 5]:
            y[i] = 0
        else:
            y[i] = 1
    y = np.array(y)
    
    documents_pd = loads_pdecks_reviews()
    # transform toy dataset labels to 0 = gluten, 1 = unknown
    y_pd = documents_pd.target
    for i in range(len(y_pd)):
        if y_pd[i] == 1 or target == 2:
            # good
            y_pd[i] = 1
        else:
            # bad
            y_pd[i] = 0
    y_pd = np.array(y_pd)

    preprocessor = PennTreebankPunkt('word2vec')

    sentences = [] # initialize an empty list of sentences
    print "Parsing sentences from training set"
    for doc in documents.data:
        sentences += preprocessor(doc)

    test_sentences = [] # initialize an empty list of sentences
    print "Parsing sentences from test set"
    for doc in documents_pd.data:
        test_sentences += preprocessor(doc)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)

    # Set values for various parameters
    num_features = 300    # Word vector dimensionality
    min_word_count = 10   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 5          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print "Training model..."
    model = word2vec.Word2Vec(sentences, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)


    #PERSIST the model
    model_name = "300features_10minwords_5context"
    model.save(model_name)

    #REVIVE the model
    # model = Word2Vec.load("300features_40minwords_10context")
    model = Word2Vec.load("300features_10minwords_5context")

    # feature vector for each word in vocab, stored in np array syn0
    # model.syn0.shape >> (737,300) == 737 300-feature-long words, 40 word min
    # model.syn0.shape >> (2182,300) == 2182 300-feature-long words, 10 word min

    clean_train_reviews = []
    for review in documents.data:
        clean_train_reviews.append( review_to_wordlist( review, remove_stopwords=True))

    trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features )

    clean_test_reviews = []
    for review in documents_pd.data:
        clean_test_reviews.append( review_to_wordlist( review, remove_stopwords=True))

    testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )


    # KMeans Clustering
    start = time.time() # Start time

    # Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
    # average of 5 words per cluster
    word_vectors = model.syn0
    num_clusters = word_vectors.shape[0] / 5

    # Initalize a k-means object and use it to extract centroids
    kmeans_clustering = KMeans( n_clusters = num_clusters )
    idx = kmeans_clustering.fit_predict( word_vectors )

    # Get the end time and print how long the process took
    end = time.time()
    elapsed = end - start
    print "Time taken for K Means clustering: ", elapsed, "seconds."
    
    # Create a Word / Index dictionary, mapping each vocabulary word to
    # a cluster number                                                                                            
    word_centroid_map = dict(zip( model.index2word, idx ))

    # For the first 10 clusters
    for cluster in xrange(0,10):
        #
        # Print the cluster number  
        print "\nCluster %d" % cluster
        #
        # Find all of the words for that cluster number, and print them out
        words = []
        for i in xrange(0,len(word_centroid_map.values())):
            if( word_centroid_map.values()[i] == cluster ):
                words.append(word_centroid_map.keys()[i])
        print words


    # now we have a cluster (or "centroid") assignment for each word,
    # and we can define a function to convert reviews into bags-of-centroids.
    # This works just like Bag of Words but uses semantically related clusters
    # instead of individual words:

    # Pre-allocate an array for the training set bags of centroids (for speed)
    train_centroids = np.zeros( (len(documents.data), num_clusters), \
        dtype="float32" )
    
    test_centroids = np.zeros( (len(documents_pd.data), num_clusters), \
        dtype="float32" )

    # Transform the training set reviews into bags of centroids
    counter = 0
    for review in clean_train_reviews:
        train_centroids[counter] = create_bag_of_centroids( review, \
            word_centroid_map )
        counter += 1
    
    counter = 0
    for review in clean_test_reviews:
        test_centroids[counter] = create_bag_of_centroids( review, \
            word_centroid_map )
        counter += 1

    forest = RandomForestClassifier(n_estimators = 100)
    forest = forest.fit(train_centroids, y)
    result = forest.predict(test_centroids)

    # compare the test results 
    comparison = zip(y_pd, result)

    for item in comparison:
        print "Target: %d\tResult: %d" % (item[0], item[1])

    return


# TODO: frequency distributions
def review_to_wordlist( review, remove_stopwords=False ):
    """
    converts a document to a sequence of words optionally
    removing stop words.  Returns a list of words.
    """
    # Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review)
    # Convert words to lower case and split them
    words = review_text.lower().split()
    # Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return(words)

def create_bag_of_centroids( wordlist, word_centroid_map ):
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max( word_centroid_map.values() ) + 1
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count 
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    # Return the "bag of centroids"
    return bag_of_centroids


def makeFeatureVec(words, model, num_features):
    """average all of the word vectors in a given paragraph"""
    
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    
    nwords = 0.
     
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.index2word)
    
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
     
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    """Given a set of reviews (each one a list of words), calculate 
    the average feature vector for each one and return a 2D numpy array"""
    counter = 0.
     
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
     
    for review in reviews:
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print "Review %d of %d" % (counter, len(reviews))
       # make average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, \
           num_features)
       
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs




def represents_int(s):
    """Helper function for checking if input string represents an int"""
    try:
        int(s)
        return True
    except ValueError:
        return False


def get_folds_and_iter():
    """
    Prompt the user for number of Kfolds (min and max) and iterations

    This information is passed to score_kfolds
    """

    # cross-validate
    # minimum K
    min_num_folds = raw_input("Enter a minimum number of folds (2-10): ")
    while not represents_int(min_num_folds):
        min_num_folds = raw_input("Enter a number of folds (2-10): ")
        print
    min_num_folds = int(min_num_folds)

    # maximum K
    if int(min_num_folds) != 10:
        max_num_folds = raw_input("Enter a maximum number of folds (%d-10): " % int(min_num_folds))
        while not represents_int(max_num_folds) or int(max_num_folds) < min_num_folds:
            max_num_folds = raw_input("Enter a maximum number of folds (%d-10): " % int(min_num_folds))
            print
    else:
        max_num_folds = 10

    # number of iterations
    num_iter = raw_input("Enter a number of iterations (1-50): ")
    print

    while not represents_int(num_iter):
        num_iter = raw_input("Enter a number of iterations (1-50): ")
        print


    max_num_folds = int(max_num_folds)
    num_iter = int(num_iter)

    return (min_num_folds, max_num_folds, num_iter)


def check_toy_dataset():
    """Checks the accuracy of the pickled classifier for the toy dataset predictions"""
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
    print "PERCENT INACCURATE: ", (inaccurate/(len(y_trans_pd)))*100.0000
    for i in range(0, len(y_trans_pd)):
        print "Index i: %s" % i
        print "Predicted: %s" % predicted_pd[i][0]
        print "Actual: %s" % str(int(y_trans_pd[i]))
        print '-'*20

    return


def train_classifier():
    """SUPERCEDED by multilabel classifier
    Trains the classifier on the labeled yelp data.

    Tests the classifier pipeline on a "new doc".

    Provides opportunities to persist the trained model and/or its components
    """
    # LOAD the training documents
    # documents = loads_pdecks_reviews(container_path, categories)
    documents = loads_yelp_reviews(container_path, categories)
    X, y = bunch_to_np(documents)

    min_num_folds, max_num_folds, num_iter = get_folds_and_iter()
    fold_avg_scores = score_kfolds(X, y, min_num_folds, max_num_folds, num_iter)

    scores = tunes_parameters(X, y, 10)

    # CREATE and TRAIN the classifier PIPELINE
    X, y = bunch_to_np(documents)
    count_vect, tfidf_transformer, clf, pipeline_clf = create_train_classifier(X, y)

    # TEST the classifier
    new_doc = ['I love gluten-free foods. This restaurant is the best.']
    new_doc_category_id_pipeline = pipeline_clf.predict(new_doc)
    new_doc_category_pipeline = get_category_name(new_doc_category_id_pipeline)

    print
    print "-- Test document --"
    print
    print "Using Pipeline:"
    print "%r => %s" % (new_doc[0], new_doc_category_pipeline)

    # PERSIST THE MODEL / COMPONENTS
    items_to_pickle = [pipeline_clf, count_vect, tfidf_transformer, clf]
    pickling_paths = [pickle_path_SVC, pickle_path_v, pickle_path_t, pickle_path_c]
    to_persist(items_to_pickle=items_to_pickle, pickling_paths=pickling_paths)

    return


def train_random_forest_classifier():
    """loads the documents from random_forest directory for multilabel classification"""
    container_path = './data/random_forest/'
    print "... Loading data from %s ..." % container_path

    documents = loads_yelp_reviews(container_path)
    X, y = bunch_to_np(documents)

    print "... Creating binary category labels ..."
    labels = ['allergy', 'gluten', 'kosher', 'paleo', 'unknown', 'vegan'] # --> [0, 1, 2, 3, 4, 5]
    binary_labels = [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]
    y_transform = []
    for i, label in enumerate(y):
        y_transform.append(binary_labels[label])

    y_transform = np.array(y_transform)

    print "... Vectorizing text ..."
    rf_vectorizer = TfidfVectorizer(strip_accents='unicode',
                                 stop_words='english',
                                 encoding='utf-8',
                                 decode_error='strict',
                                 ngram_range=(1, 1),
                                 preprocessor=PennTreebankPunkt())

    X_tfidf = rf_vectorizer.fit_transform(X)

    # Initialize a random forest with 100 trees
    forest_clf = OneVsRestClassifier(RandomForestClassifier(n_estimators = 100))

    # fit the forest to the training set, using the tf-idf as features
    # and the category labels as the response variable
    print "... Training the Random Forest classifier ..."
    forest_clf = forest_clf.fit(X_tfidf, y_transform)

    print "... Training complete!"
    print

    sample_text = "From start to finish, the meal was perfection. My gluten-loving significant other and I both ordered the tasting menu, and our server assured me with confidence that they could make substitutions for a few of the items. They brought out some amuse bouche -- even GF versions for me(!); they gave me delicious GF bread (!!); and they even had tasty GF pasta that was close to the real thing. I never felt like I was missing out or that I was a burden. It was a relaxed affair, and I am so grateful to the entire staff's commitment to excellent service for all diners."

    X_sample_tfidf = rf_vectorizer.transform([sample_text])

    sample_predict = forest_clf.predict(X_sample_tfidf)
    print
    print "-- Sample text prediction --"
    print sample_text
    print "Predicted categories: "
    for predict_list in sample_predict:
        for i, label in enumerate(predict_list):
            if label == 1:
                print labels[i]
    print

    sample_text2 = 'I love vegan and gluten-free foods vegan vegan gluten GF vegan'
    X_sample_tfidf2 = rf_vectorizer.transform([sample_text2])
    sample_predict2 = forest_clf.predict(X_sample_tfidf2)
    print
    print "-- Sample text prediction --"
    print sample_text2
    print "Predicted categories: "
    for predict_list in sample_predict2:
        for i, label in enumerate(predict_list):
            if label == 1:
                print labels[i]
    print

    # PERSIST THE MODEL / COMPONENTS
    # TODO: do I need to pickle the vectorizer used here??
    items_to_pickle = [rf_vectorizer, forest_clf]
    pickling_paths = [pickle_path_rfv, pickle_path_rfc]
    to_persist(items_to_pickle=items_to_pickle, pickling_paths=pickling_paths)

    return forest_clf


def categorize_text(text, revive=True):
    """
    For a text, classify it with multilabel classifier (random forest)
    return an array of label categories ('gltn', etc.)

    >>> text = "This gluten-free restaurant also has great vegan and paleo options."
    >>> predictions = categorize_text(text)

    -- Text prediction --
    [[0 0 0 0 0 0]]

    >>> text = 'I love vegan and gluten-free foods vegan vegan gluten GF vegan'
    >>> predictions = categorize_text(text)

    -- Text prediction --
    [[0 1 0 0 0 0]]
    Predicted categories:
    gltn


    This shows that even though the restaurant was categorized as 'good' (1),
    the probability that it is good is only 0.56, which is almost neutral.
    """
    labels = ['algy', 'gltn', 'kshr', 'pleo', 'unkn', 'vgan']

    if not isinstance(text, (np.ndarray, np.generic) ):
        if isinstance(text, list):
            text = np.array(text)
        else:
            text = np.array([text])

    # TODO: keep pickle_paths in list
    prediction_list = []
    if revive == True:
        # revive vectorizer
        vectorizer = revives_component(pickle_path_rfv)
        text_tfidf = vectorizer.transform(text)
        # revive classifier
        forest_clf = revives_component(pickle_path_rfc)
        text_predict = forest_clf.predict(text_tfidf)
        print
        print "-- Text prediction --"
        print text_predict
        print "Predicted categories: "
        for predict_list in text_predict:
            for i, label in enumerate(predict_list):
                if label == 1:
                    print labels[i]
                    prediction_list.append(labels[i])
    # if the text does not fit in any category, classify it as unknown
    if not prediction_list:
        prediction_list = ['unkn']

    return prediction_list


def random_forest_training_set():
    """Makes a copy of the files in /data/keywords/subfolder for use with random forest classifier"""
    from os import listdir
    import codecs

    old_path = './data/keywords/'
    new_path = './data/random_forest/'
    categories = ['unknown', 'gluten', 'allergy', 'paleo', 'kosher', 'vegan']
    for cat in categories:
        # open file in old path, copy to new path, strip out info ...

        # get list of document name
        container_path = old_path + cat
        new_container_path = new_path + cat
        documents = [d for d in sorted(listdir(container_path))]
        for d in documents:
            file_path = container_path + '/' + d
            with codecs.open(file_path, 'r', 'utf-8-sig') as f:
                file_text = f.read()
                file_text = file_text.split('|')[4]
                f.close()
            new_file_path = new_container_path + '/' + d
            with codecs.open(new_file_path, 'w', 'utf-8-sig') as f:
                f.write(file_text)
                f.close()


if __name__ == "__main__":

    ## TRAIN SENTIMENT ANALYSIS CLASSIFIER
    to_test = raw_input("Train the gluten-free sentiment analysis classifier? Y or N >>")
    if to_test.lower() == 'y':
        train_sentiment_classifier()

    ## RANDOM FOREST MULTILABEL PROBLEM
    print
    to_test = raw_input("Train the random forest multilabel classifier? Y or N >>")
    if to_test.lower() == 'y':
        forest_clf = train_random_forest_classifier()

    ## TEST SENTIMENT ANALYSIS PROTOTYPE AND PLOT METRICS
    print
    to_test = raw_input("Train the gluten-free sentiment analysis vectorizer? Y or N >>")
    if to_test.lower() == 'y':
        # LOAD DATASET
        data_choice = raw_input("Enter a dataset to classify: [P]decks, [Y]elp, [S]tars >> ")

        while data_choice.lower() not in ['y', 'p', 's']:
            data_choice = raw_input("Enter a dataset to classify: [P]decks, [Y]elp >> ")

        # FEATURE EXTRACTION
        if data_choice.lower() == 'y':
            vectorizer, all_avg_sentiment_scores = train_sentiment_vectorizer(dataset='yelp')
        if data_choice.lower() == 's':
            vectorizer, all_avg_sentiment_scores = train_sentiment_vectorizer(dataset='stars')
        else:
            vectorizer, all_avg_sentiment_scores = train_sentiment_vectorizer()

        # PLOT RESULTS
        user_choice = ''
        while user_choice.lower() not in ['y', 'n']:
            user_choice = raw_input("Would you like to plot the cross-validation results? Y or N >> ")

        if user_choice.lower() == 'y':
            mean_scores_by_kfolds = plot_sentiment_model_scores(all_avg_sentiment_scores)

        # PERSIST THE VECTORIZER
        to_persist(items_to_pickle=[vectorizer], pickling_paths=[pickle_path_SA_v])

    ## USE SENTIMENT ANALYSIS PROTOTYPE
    print
    to_test = raw_input("Perform sentiment analysis? Y or N >>")
    if to_test.lower() == 'y':
        text = raw_input("Enter a sentence to parse >> ")
        target = predict_sentiment(text, categories=['gltn'])
        print "Target category: %s" % target[0][0]
        if target[0][1] == 1:
            print "Prediction: good"
        else:
            print "Prediction: bad"
        print "Probability of good:" % target[0][2]


    ## SUPERCEDED CLASSIFIERS ##
    ## TRAIN AND PERSIST CLASSIFIER ## SUPERCEDED -- USE RANDOM FORESTS
    print
    to_train = raw_input("Train the LinearSVC classifier for categorization? Y or N >> ")
    if to_train.lower() == 'y':
        train_classifier()

    ## CHECK PERFORMANCE OF PICKLED CLASSIFIER ON TOY DATA SET ## SUPERCEDED
    else:
        print
        to_test = raw_input("Check the pipeline classifier on the toy data set? Y or N >>")
        if to_test.lower() == 'y':
            check_toy_dataset()
