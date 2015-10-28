## LOADING THE 20 NEWSGROUPS DATASET ##

# to achieve faster execution times for this first example, work with only 4
# categories out of the 20 available in the newsgroups dataset
categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']

# load the list of files matching those categories
from sklearn.datasets import fetch_20newsgroups

# returns a scikit-learn 'bunch': object with fields that can be accessed
# as python dict keys or object attributes
# NOTE: this part takes several minutes to fetch the data ...
twenty_train = fetch_20newsgroups(subset='train',
                                  categories=categories,
                                  shuffle=True,
                                  random_state=42
                                  )

# list of the requested category names
twenty_train.target_names

# files are loaded in memory in data attribute
len(twenty_train.data)  # 2257

# filenames also available
len(twenty_train.filenames)  # 2257

# print the first three lines of the first loaded filenames
print "\n".join(twenty_train.data[0].split("\n")[:3])

# category is the name of the newsgroup
# sklearn loads .target as an array of integers that correspond to the indices
# of the category names in target_names list
print twenty_train.target_names[twenty_train.target[0]]

twenty_train.target[:10]

# RESULTS:
# array([1, 1, 3, 3, 3, 3, 3, 2, 2, 2])

# retrieve the category names
for t in twenty_train.target[:10]:
    print (twenty_train.target_names[t])

# RESULTS:
# comp.graphics
# comp.graphics
# soc.religion.christian
# soc.religion.christian
# soc.religion.christian
# soc.religion.christian
# soc.religion.christian
# sci.med
# sci.med
# sci.med


## EXTRACTING FEATURES FROM TEXT FILES ##


# TOKENIZING TEXT
# build a dictionary of features and transform documents to feature vectors

from sklearn.feature_extraction.text import CountVectorizer

# Create an instance of CountVectorize feature extractor
# CountVectorize supports counts of N-grams of words or consecutive characters.
count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape  # (2257, 35788)

# once fitted, the vectorizer has built a dictionary of feature indices
# the index value of a word in the vocabulary is linked to its frequency 
# in the whole training corpus
count_vect.vocabulary_.get('algorithm')  # 4690
count_vect.vocabulary_.get('pretty')  # 25972
count_vect.vocabulary_.get('heinous')  # 16394
count_vect.vocabulary_.get('the')  # 32142


# FROM OCCURRENCES TO FREQUENCIES: TF-IDF
# perform tf and tf-idf --> transform to feature vectors

from sklearn.feature_extraction.text import TfidfTransformer

# create an instance of TfidfTransformer, tf only, fit to X_train_counts
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)

# perform the transformation on the frequency counts fit
X_train_tf = tf_transformer.transform(X_train_counts)

X_train_tf.shape # (2257, 35788)


# create an instance of TfidfTransformer, tf & idf, fit to X_train_counts
tfidf_transformer = TfidfTransformer()

# combines fit(..) and transform(..) with fit_transform(..)
# perform the transformation on the frequency counts fit
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

X_train_tfidf.shape # (2257, 35788)



# TRAINING A CLASSIFIER
# start with a naive Bayes classifier, specifically a multinomial variant

from sklearn.naive_bayes import MultinomialNB

# define multinomial naive Bayes classifier
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

# define new documents
docs_new = ['God is love', 'OpenGL on the GPU is fast']

# extract the features from the new document
# call with transform, not fit_transform)
X_new_counts = count_vect.transform(docs_new)

# apply tf-idf transformation
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

# predict class of new document (returns index value of category name)
predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print '%r => %s' % (doc, twenty_train.target_names[category])



# BUILDING A PIPELINE
# Pipeline class behaves like a compound classifier to make 
# vectorizer => transformer => classifier easier to work with

from sklearn.pipeline import Pipeline

# Pipeline([ (vectorizer), (transformer), (classifier)])
# names 'vect', 'tfidf', and 'clf' are arbitrary (chosen)
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
])

# train the model with a single command
# .fit(data, classes)
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)



# EVALUATE PERFORMANCE ON TEST SET
import numpy as np
twenty_test = fetch_20newsgroups(subset='test',
                                 categories=categories,
                                 shuffle=True,
                                 random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
np.mean(predicted == twenty_test.target)  # 0.8348868175... (83.4% accuracy)


# CHANGE THE LEARNER by plugging in a different classifier into pipeline
from sklearn.linear_model import SGDClassifier

# note penalty = 'l2' (with a lowercase L)
txt_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, n_iter=5,
                                           random_state=42))
])

# train the linear SVM model
txt_clf = txt_clf.fit(twenty_train.data, twenty_train.target)
predicted = txt_clf.predict(docs_test)
np.mean(predicted == twenty_test.target)  # 0.9127829... (91.2% accuracy)


from sklearn import metrics
print metrics.classification_report(twenty_test.target, predicted,
                                    target_names=twenty_test.target_names)

# RESULTS
#                         precision    recall  f1-score   support

#            alt.atheism       0.95      0.81      0.87       319
#          comp.graphics       0.88      0.97      0.92       389
#                sci.med       0.94      0.90      0.92       396
# soc.religion.christian       0.90      0.95      0.93       398

#            avg / total       0.92      0.91      0.91      1502
# END RESULTS


## PARAMETER TUNING USING GRID SEARCH
# Instead of tweaking the parameters of the various components of the chain,
# it is possible to run an exhaustive search of the best parameters on a grid
# of possible values

from sklearn.grid_search import GridSearchCV

# try all classifiers on either words or bigrams, with or without idf,
# and with a penalty parameter of either 0.01 or 0.001 for the linear SVM
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3),
             }

# define grid search on SVM classifier (txt_clf)
gs_clf = GridSearchCV(txt_clf, parameters, n_jobs=-1)

# perform search on smaller subset of training data
# the result of calling fit on a GridSEarchCV object is a classifier that we
# can use to predict
gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])

# for the doc 'God is love', make a prediction and find category name in index
twenty_train.target_names[gs_clf.predict(['God is love'])]

# retrieve optimal parameters by inspecting the object's grid_scores attribute
# list of parameters/score pairs
# WTF IS GOING ON WITH LINE 225?????
best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
for param_name in sorted(parameters.keys()):
    print "%s: %r" % (param_name, best_parameters[param_name])

# clf__alpha: 0.001
# tfidf__use_idf: True
# vect__ngram_range: (1, 1)

score # 0.900...
