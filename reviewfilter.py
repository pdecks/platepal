"""Loops over all entries (here Yelp reviews) and uses the classifier to get
a best guess at the classification. It shows this best guess to the user and
then asks what the correct category should have been. When run with a new
classifier, the guesses will at first be random, but they should improve
over time.

>>> import docclass
>>> import reviewfilter as rf
>>> c1 = docclass.FisherClassifier(docclass.getwords)
>>> c1.setdb('pdecks_reviews.db')
>>> my_dir = '/Users/pdecks/hackbright/project/Yelp/pdecks-reviews/'
>>> filelist = rf.generate_filelist(my_dir)
>>> my_reviews = rf.generate_review_dict(filelist)

by: Patricia Decker, modified from Programming Collective Intelligence
date: 10/26/2015
"""

# TODO: CONSIDER NEURAL NETWORK? can capture interdepence of features but
# requires significantly more computing power and offers less clarity into
# how each feature contributes to the final score

import os
import glob
import re

my_dir = '/Users/pdecks/hackbright/project/Yelp/pdecks-reviews/'

def generate_filelist(my_dir):
    """Takes a directory path and returns a list of text files in directory."""
    filelist = []
    os.chdir(my_dir)
    for files in glob.glob("*.txt"):
        filelist.append(files)
    return filelist

def generate_review_dict(filename):
    """Given a filename, make a dictionary entry"""
    review_dict = {}

    review_file = open(filename)
    review_data = review_file.read()

    # extract review data, splitting on pipes
    current_review = review_data.split("|")

    # assign meaningful variable names
    rest_name = current_review[0]
    rest_stars = current_review[1]
    review_date = current_review[2]
    review_text = current_review[3]

    # reformat date to match Yelp JSON
    review_date_month = review_date[:2]
    review_date_day = review_date[2:4]
    review_date_year = review_date[4:]
    yelp_date = review_date_year + '-' + review_date_month + '-' + review_date_day

    # create empty votes dictionary (not available in text files)
    vote_dict = {'useful': 0, 'funny': 0, 'cool': 0}

    # assign values to dictionary keys
    review_dict = {'type': 'review',
                   'business_id': rest_name,
                   'user_id': 'greyhoundmama',
                   'stars': rest_stars,
                   'text': review_text,
                   'date': yelp_date,
                   'votes': vote_dict
                   }

    review_file.close()

    return review_dict

def generate_reviews_dict(filelist):
    """Takes a list of files (reviews) and returns a list of reviews dictionaries."""
    reviews_list = []

    for review in filelist:
        # make the dictionary entry for current review
        review_dict = generate_review_dict(review)

        # add to larger review dictionary
        reviews_list.append(review_dict)

    return reviews_list

# TODO: add check for word already in dictionary
def define_entry_features(entry_text):
    """Feature extractor.

    Takes review text (entry text) and splits it on any sequence of non-alphanumeric
    characters. Here, the resulting words and word pairs are the document features.

    Also flags a review as having too much shouting based on uppercase words.
    """
    splitter = re.compile('\\W*')
    # print "This is splitter: %s" % splitter
    # print "This is entry_text:\n%s" % entry_text
    f = {}

    # Extract the summary words
    # summary_words = []
    # for s in splitter.split(entry_text):
    #     if len(s) > 2 and len(2) < 20:
    #         summary_words.append(s.lower())
    summary_words = [s.lower() for s in splitter.split(entry_text)
                     if len(s) > 2 and len(s) < 20]

    # Count uppercase words
    uc = 0
    for i in range(len(summary_words)):
        w = summary_words[i]
        f[w] = 1
        if w.isupper():
            uc += 1

        # get word pairs in summary as features
        if i <= len(summary_words)-2:
            if i == len(summary_words)-2:
                two_words = ' '.join(summary_words[i:])
            else:
                two_words = ' '.join(summary_words[i:i+2])
            f[two_words] = 1

    # UPPERCASE is a virtual word flagging too much shouting
    if float(uc) / len(summary_words) > 0.3:
        f['UPPERCASE'] = 1

    return f



def classify_reviews(review_list, classifier):
    """Takes a list of review dictionaries and classifies the entries."""

    for entry in review_list:

        print '-' * 60
        print "Business name: %s" % entry['business_id']
        print "User: %s" % entry['user_id']
        print "Stars: %s" % entry['stars']
        print "Review date: %s" % entry['date']
        print "Review: %s" % entry['text']
        print "Votes: useful %s, funny %s, cool %s" % (entry['votes']['useful'],
                                                       entry['votes']['funny'],
                                                       entry['votes']['cool']
                                                       )
        print '-' * 60

        # text to be classified
        fulltext = entry['text']

        # print the best guess at the current category
        print 'Guess: ' + str(classifier.classify(fulltext))

        # Ask the user to specify the correct category and train on that
        print "Rate for gluten-free safety as one of the following:"
        print "'Excellent', 'Good', 'Neutral', 'Limited', 'Shady', 'Bad'"
        user_cat = raw_input('Enter category: ')

        # validate user entry
        classifications = ['excellent', 'good', 'neutral', 'limited', 'shady', 'bad']
        while user_cat.lower() not in classifications:
            print "Rate for gluten-free safety as one of the following:"
            print "'Excellent', 'Good', 'Neutral', 'Limited', 'Shady', 'Bad'"
            user_cat = raw_input('Enter category: ')

        classifier.train(fulltext, user_cat.lower())

# ALT DATA STRUCTURE USING DICT OF DICT ##
# def classify_reviews(review_dict, classifier):
#     """Takes a dictionary of reviews and classifies the entries."""
#     print "In classify_reviews ..."
#     for entry in review_dict:

#         print '-' * 60
#         print "Business name: %s" % review_dict[entry]['name']
#         print "Score: %s" % review_dict[entry]['score']
#         print "Review date: %s" % review_dict[entry]['date']
#         print "Review: %s" % review_dict[entry]['review']
#         print '-' * 60

#         # combine all the text to create one item for the classifier
#         fulltext = '%s\n%s\n%s\n%s' % (review_dict[entry]['name'],
#                                        review_dict[entry]['score'],
#                                        review_dict[entry]['date'],
#                                        review_dict[entry]['review'])
#         # print fulltext

#         # print the best guess at the current category
#         print 'Guess: ' + str(classifier.classify(fulltext))

#         # Ask the user to specify the correct category and train on that
#         user_cat = raw_input('Enter category: ')
#         classifier.train(fulltext, user_cat)


if __name__ == "__main__":
    filelist = generate_filelist(my_dir)
    my_reviews = generate_reviews_dict(filelist)
    print '\n\n'
    print "RESTAURANTS REVIEWED"
    for restaurant in my_reviews.keys():
        print restaurant
    print '\n\n'
    print '-'*60

    for review in my_reviews:
        print "%s" % review
        print my_reviews[review]
        print '-'*60

# import docclass
# import reviewfilter as rf
# c1 = docclass.FisherClassifier(docclass.getwords)
# c1.setdb('pdecks_reviews.db')
# my_dir = '/Users/pdecks/hackbright/project/Yelp/pdecks-reviews/'
# filelist = rf.generate_filelist(my_dir)
# my_reviews = rf.generate_reviews_dict(filelist)
# rf.classify_reviews(my_reviews, c1)


## DEBUGGING

# print "This is entry['name']: %s" % review_dict[entry]['name']
# print "This is entry['score']: %s" % review_dict[entry]['score']
# print "This is entry['date']: %s" % review_dict[entry]['date']
