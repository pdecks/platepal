"""
Learns how to classify a document by being trained. Accuracy improves
with continued exposure to new information as classifier learns which
features are important for making a distinction.

from Programming Collective Intelligence, by Toby Segaran.
"""

import re
import math
import sqlite3


def sampletrain(c1):
    """Dumps some sample training data into the classifier."""
    c1.train('Nobody owns the water.', 'good')
    c1.train('the quick rabbit jumps fences', 'good')
    c1.train('buy pharmaceuticals now', 'bad')
    c1.train('make quick money at the online casino', 'bad')
    c1.train('the quick brown fox jumps', 'good')


def getwords(doc):
    """Feature extractor.

    Takes a document and splits it on any sequence of non-alphanumeric
    characters. Here, the resulting words are the document features.
    """

    # compiling regular expressions with python built-in library re, a C
    # extension module. Note REs are handled as strings because REs are not
    # part of core Python so no syntax was created for expressing them.

    # RegEx are compiled into PATTERN OBJECTS, thus splitter is a pattern obj
    # note compile('\\W*') is the string literal form
    # the raw string form would be compile(r'\W*')
    splitter = re.compile('\\W*')

    # split the words by non-alpha characters
    # the split() method OF A PATTERN splits a string apart whever the RegEx
    # matches, returning a list of the pieces.
    # below command ignores 2-letter words and words longer than 20 letters
    words = [s.lower() for s in splitter.split(doc)
             if len(s) > 2 and len(s) < 20]

    # return the unique set of words only
    # TODO: weight words that appear multiple times?
    return dict([(w, 1) for w in words])


class Classifier(object):
    """Encapsulate what the classifier has learned so far. This allows for
    instantiation of multiple classifiers for different users, groups, or
    queries that can be trained to a particular group's needs."""

    def __init__(self, getfeatures, filename=None):
        # Classifier.__init__(self, getfeatures) ## WHY DO WE NEED THIS?!?!?!
        # Counts of feature/category combinations
        # ex. {'python': {'bad': 0, 'good': 6}, 'the': {'bad': 3, 'good': 3}}
        # where 'python' is a feature and 'bad' and 'good' are categories
        self.fc = {}

        # Counts of documents in each category
        # dict of how many times every classification has been used
        # this is needed for probability calculations
        # ex. # documents labeled 'good' or 'bad' --> {'good': 3, 'bad': 3}
        self.cc = {}

        # Function that will be used to extract the features from items
        # to be classified (ex: getwords)
        self.getfeatures = getfeatures

        # for deciding in which category a new item belongs
        self.thresholds = {}


    ## CREATE HELPER METHODS TO INCREMENT AND GET THE COUNTS ##
    # note that .setdefault() will set d[key]=default IF key not already in d
    # here, if feature not yet in dict, create key
    # then, for each feature dictionary, if category not yet in feature dict,
    # add category as value and set it to 0. THEN, for all cases, increment.

    # def incf(self, f, cat):
    #     """Increases the count of a feature/category pair."""
    #     self.fc.setdefault(f,{})
    #     self.fc[f].setdefault(cat, 0)
    #     self.fc[f][cat] += 1


    # def incc(self, cat):
    #     """Increase the count of a document category (classification)."""
    #     self.cc.setdefault(cat, 0)
    #     self.cc[cat] += 1


    # def fcount(self, f, cat):
    #     """Returns a float of no. times a feature has appeared in a category."""
    #     if f in self.fc and cat in self.fc[f]:
    #         return float(self.fc[f][cat])
    #     return 0.0


    # def catcount(self, cat):
    #     """Returns a float of no. times a category occurs as a classification."""
    #     if cat in self.cc:
    #         return float(self.cc[cat])
    #     return 0


    # def totalcount(self):
    #     """Returns total number of all category (classification) occurrences."""
    #     return sum(self.cc.values())


    # def categories(self):
    #     """Returns list of all categories (classifications)."""
    #     return self.cc.keys()


    ## CREATE HELPER METHODS: DATABASE VERSIONS ##

    def incf(self, f, cat):
        """Increases the count of a feature/category pair."""
        count = self.fcount(f,cat)
        # if feature doesn't exist, insert it into table
        if count == 0:
            self.con.execute("insert into fc values ('%s', '%s', 1)"
                             % (f, cat))
        else: # update value in table
            self.con.execute(
                "update fc set count=%d where feature='%s' and category='%s'"
                % (count + 1, f, cat))
    
    def incc(self, cat):
        """Increase the count of a document category (classification)."""
        count = self.catcount(cat)
        if count == 0:
            self.con.execute("insert into cc values ('%s', 1)" % (cat))
        else:
            self.con.execute("update cc set count=%d where category='%s'"
                             % (count + 1, cat))

    def fcount(self, f, cat):
        """Returns a float of no. times a feature has appeared in a category."""
        res = self.con.execute(
            'select count from fc where feature="%s" and category="%s"'
            % (f,cat)).fetchone()
        if res == None:
            return 0
        else:
            return float(res[0])

    def catcount(self, cat):
        """Returns a float of no. times a category occurs as a classification."""
        res = self.con.execute('select count from cc where category="%s"'
                               %(cat)).fetchone()
        if res == None:
            return 0
        else:
            return float(res[0])

    def totalcount(self):
        """Returns total number of all category (classification) occurrences."""
        res = self.con.execute('select sum(count) from cc').fetchone()
        if res == None:
            return 0
        else:
            return res[0]

    def categories(self):
        """Returns list of all categories (classifications)."""
        cur = self.con.execute('select category from cc')
        return [d[0] for d in cur]


    ## TRAINING THE MODEL ##

    def train(self, item, cat):
        """Takes an item (e.g., document) and a classification.
        Increments counters."""

        # extract the features
        features = self.getfeatures(item)

        # increment the count for every feature with this category
        for f in features:
            self.incf(f, cat)

        # increment the count for this category
        self.incc(cat)

        ## FOR PERSISTING IN DATABASE ##
        if self.con:
            self.con.commit()


    ## CALCULATING PROBABILITIES ##

    def fprob(self, f, cat):
        """Returns conditional probability P(A|B) = P(word|classification).

        >>> import docclass
        >>> c1 = docclass.Classifier(docclass.getwords)
        >>> docclass.sampletrain(c1)
        >>> c1.fprob('quick', 'good')
        0.6666666666666666
        """
        if self.catcount(cat) == 0: return 0

        # the total number of times this feature appeared in this category
        # divided by the total number of items in this category
        return self.fcount(f, cat) / self.catcount(cat)


    def weightedprob(self, f, cat, prf, weight=1.0, ap=0.5):
        """Returns a weighted average of getprobability and assumed
        probability.

        prf = probability function
        ap = assumed probability
        """

        # calculate current probability
        basicprob = prf(f, cat)

        # count the number of times this feature has appeared in all categories
        totals = sum([self.fcount(f, c) for c in self.categories()])

        # calculate the weighted average
        bp = ((weight * ap) + (totals * basicprob)) / (weight + totals)

        return bp

    ## CHOOSING A CATEGORY ##

    def setthreshold(self, cat, t=1.0):
        """Sets a minimum threshold for each category."""
        self.thresholds[cat] = t

    def getthreshold(self, cat):
        """Retuurns the threshold for a category."""
        if cat not in self.thresholds:
            return 1.0
        return self.thresholds[cat]

    def classify(self, item, default='unknown'):
        """Calculates the probability for each category. Determines which one 
        is the largest and whether it exceeds the next largest by more than
        its threshold. If none of the categories can accomplish this, the
        method just returns the default values."""

        probs = {}

        # Find the category with the highest probability
        max = 0.0
        for cat in self.categories():
            probs[cat] = self.prob(item, cat) # note that self.prob on subclass ...
            if probs[cat] > max:
                max = probs[cat]
                best = cat

        # ensure the probability exceeds threshold * next_best
        for cat in probs:
            if cat == best:
                continue
            if probs[cat] * self.getthreshold(best) > probs[best]:
                return default
            return best


    ## PERSISTING TRAINED CLASSIFIERS ##

    def setdb(self, dbfile):
        """Opens a database for this classifier and creates tables, if necessary."""
        self.con = sqlite3.connect(dbfile)
        self.con.execute('create table if not exists fc(feature, category, count)')
        self.con.execute('create table if not exists cc(category, count)')


## SUBCLASSES ################################################################

# Once you have the probabilities of a document in a category containing a
# particular word, you need a way to combine the individual word probabilities
# to get the probability that an entire document belongs in a given category.

# naive bayesian -- "naive" assumes the probabilities being combined are
# independent of each other. this is a false assumption, so we can't actually
# use the probability from the naive classifier as the actual probability
# that a document belongs in that category. BUT we can still compare results
# for different categories and see which one has the highest probability.

# calculating the entire document probability is a matter of multiplying
# together all the probabilities of the individual words in that document.

class NaiveBayes(Classifier):
    """Subclass of Classifier for calculating the entire document probability.
    """

    def __init__(self, getfeatures, filename=None):
        """Inherit Classifier __init__"""
        return super(NaiveBayes, self).__init__(getfeatures, filename)

    def docprob(self, item, cat):
        """Extracts the features and returns an overall probability of a
        document being classified as a particular category.

        Returns P(Document | Category)"""

        features = self.getfeatures(item)

        # multiply the probabilities of all the features together
        p = 1
        for f in features:
            p *= self.weightedprob(f, cat, self.fprob)
        return p

    def prob(self, item, cat):
        """Calculates P(Category) and returns P(Doc | Cat) * P(Cat).

        Used with Bayes' Theorem to calatulate P(Category | Document)."""

        catprob = self.catcount(cat) / self.totalcount()
        docprob = self.docprob(item, cat)
        return docprob * catprob


class FisherClassifier(Classifier): ## DOUBLE CHECK CODE HERE
    """Fisher showed that is the probabilities were independent and random,
    the result of the fisherprob calculation fit a chi-squared distribution.

    An item that doesn't belong in a category should have words of varying
    feature probabilities for that category (appears random).

    And item that does belong in a category should have many features with
    high probabilities.

    By feeding the result of the Fisher calculation to the inverse chi-square
    function, you get the probability that a random set of probabilities would
    return such a high number."""
    def __init__(self, getfeatures, filename=None):
        """Inherit Classifier __init__ and create threshold dict"""
        super(FisherClassifier, self).__init__(getfeatures, filename)
        self.minimums = {}

    # def __init__(self, getfeatures):
    #     Classifier.__init__(self, getfeatures) ## WHY???!?!
    #     # create dictionary for storing classification thresholds
    #     self.minimums = {}

    def setminimum(self, cat, min):
        """Sets minimum threshold for category classification."""
        self.minimums[cat] = min

    def getminimum(self, cat):
        """Retrieve minimum threshold."""
        if cat not in self.minimums:
            return 0
        return self.minimums[cat]

    def classify(self, item, default=None):
        """Calculate the probabilities for each category and returns the
        best result that exceeds the specified minimum."""

        # loop through looking for the best result
        best = default
        maxprob = 0.0
        for c in self.categories():
            p = self.fisherprob(item, c)
            # check if it exceeds its threshold
            if p > self.getminimum(c) and p > maxprob:
                best = c
                maxprob = p
        return best

    def cprob(self, f, cat):
        """
        Returns the probability that an item with the specified feature
        belongs in the specified category, assuming there will be an equal no.
        of items in each category.

        Returns P(category | feature)

        cprob = clf / (clf + nclf) = clf / freqsum
        clf = P(feature | category) for this category
        freqsum = Sum of P(feature | category) for all categories
        """
        # The frequency of this feature in this category
        clf = self.fprob(f, cat)
        if clf == 0:
            return 0

        # the frequency of this feature in all the categories
        freqsum = sum([self.fprob(f, c) for c in self.categories()])

        # probability = freq in this cat / overall frequency
        p = clf / (freqsum)

        return p

    def fisherprob(self, item, cat):
        """fisher probability"""
        # multiply all the probabilities together
        p = 1
        features = self.getfeatures(item)
        for f in features:
            p *= (self.weightedprob(f, cat, self.cprob))

        # take the natural log and multiply by -2
        fscore = -2 * math.log(p)

        # use the inverse chi2 function to get a probability
        return self.invchi2(fscore, len(features) * 2)

    def invchi2(self, chi, df):
        """Inverse chi-square function """
        m = chi / 2.0
        chisum = term = math.exp(-m)
        for i in range(1, df//2):
            term *= m / i
            chisum += term
        return min(chisum, 1.0)





###### TESTING ###############################################################

## Test the class using python interactively ##
# import docclass
# c1 = docclass.Classifier(docclass.getwords)
# docclass.sampletrain(c1)
# c1.fcount('quick', 'good') --> returns 1.0
# c1.fcount('quick', 'bad') --> returns 1.0


# ONCE the Naive Bayes classifier is built, test with the following:
# reload(docclass) # or import docclass
# c1 = docclass.NaiveBayes(docclass.getwords)
# docclass.sampletrain(c1)
# c1.prob('quick rabbit', 'good')
# c1.prob('quick rabbit', 'bad')
## RESULT: based on the training data, the phrase 'quick rabbit' is considered
## a much better candidate for the good category than the bad


# ONCE the thresholds have been added:
# reload(docclass)
# c1 = docclass.NaiveBayes(docclass.getwords)
# docclass.sampletrain(c1)
# c1.classify('quick rabbit')
# c1.classify('quick money')
# c1.setthreshold('bad', 3.0)
# c1.classify('quick money')
# for i in range(10): docclass.sampletrain(c1)
# c1,classify('quick money')

# FISHER test
# reload(docclass)
# c1 = docclass.FisherClassifier(docclass.getwords)
# docclass.sampletrain(c1)
# c1.cprob('quick', 'good')
# c1.cprob('money', 'bad')
# c1.weightedprob('money', 'bad', c1.cprob)

# Fisher with fisherclassifier
# reload(docclass)
# c1 = docclass.FisherClassifier(docclass.getwords)
# docclass.sampletrain(c1)
# c1.cprob('quick', 'good')
# c1.fisherprob('quick rabbit', 'good')
# c1.fisherprob('quick rabbit', 'bad')

# Fisher with thresholds
# reload(docclass)
# c1 = docclass.FisherClassifier(docclass.getwords)
# docclass.sampletrain(c1)
# c1.classify('quick rabbit')
# c1.classify('quick money')
# c1.setminimum('bad', 0.8)
# c1.classify('quick money') 
## NOTE: c1.fisherprob('quick money', 'bad') --> 0.701 < 0.8 
## therefore classified as 'good'
# c1.setminimum('good', 0.4)
# c1.classify('quick money')