"""
1. Extract a set of reviews from the database using keywords.
2. Populate entries in ReviewCategory table (revcats)
    revcat_id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    review_id = db.Column(db.Integer, db.ForeignKey('reviews.review_id'))
    biz_id = db.Column(db.Integer, db.ForeignKey('biz.biz_id'))
    cat_code = db.Column(db.Integer, db.ForeignKey('categories.cat_code'))
    sen_score = db.Column(db.Float, nullable=True)  # machine generated score
    user_sen = db.Column(db.Float, nullable=True)  # for user feedback on score

by Patricia Decker 11/2015, part of Hackbright Project
"""

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import not_

import datetime

from model import YelpBiz, YelpUser, YelpReview
from model import PlatePalBiz, PlatePalUser, PlatePalReview
from model import UserList, ListEntry
from model import Category, ReviewCategory, BizSentiment
from model import Sentence, SentenceCategory
from model import connect_to_db, db

from seed import seed_revcat

from server import app

import os
import codecs
import random
import math

# absolute dir the script is in
script_dir = os.path.dirname(__file__)


def get_category_and_keywords():
    """Define category name and list of case-sensitive keywords."""
    # ex. cat_name = "unknown" or "gluten"

    print 'Select set to test or train:'
    print '0. Unrestricted'
    print '1. Gluten-free'
    print '2. Allergies'
    print '3. Paleo'
    print '4. Kosher'
    print '5. Vegan'

    user_input = raw_input(">> ")
    cat_names = ['unknown', 'gluten', 'allergy', 'paleo', 'kosher', 'vegan']

    # check input and recurse if needed
    if represents_int(user_input):
        cat_name = cat_names[int(user_input)]
    else:
        print
        print "Incorrect value entered. Please enter a number from 0-5."
        get_category_and_keywords()

    # search_terms = ["gluten", "GF", "celiac", "gluten-free"]
    if user_input == 0:
        print "UNRESTRICTED selected."
        print
        print "Enter a keyword to be excluded from search."

    else:
        print "%s selected." % cat_name.upper()
        print
        print "Enter a keyword for the category: "

    print "Keywords are case sensitive."
    keyword = raw_input(" (press enter if done) >> ")
    search_terms = []
    while keyword:
        search_terms.append(keyword)
        keyword = raw_input(" (press enter if done) >> ")

    print
    max_results = raw_input("Enter the maximum number of results to return >> ")

    if not represents_int(max_results):
        print "Please enter an integer."
        max_results = raw_input("Enter the maximum number of results to return >> ")

    max_results = int(max_results)

    return cat_name, search_terms, max_results


def get_category_reviews(cat_name, search_terms, max_results=1000):
    """Queries the DB for all reviews containing a word from the category"""

    if cat_name == "unknown":
        not_str = 'WHERE'
        j = 1
        for cname in search_terms:
            nstr = '"%' + cname + '%"'

            if j != len(search_terms):
                end_str = ' AND'
            else:
                end_str = ";"

            not_str = not_str + ' Reviews.text NOT LIKE ' + nstr + end_str
            j += 1

        QUERY =  """
        SELECT Biz.biz_id, Biz.name, Reviews.yelp_stars, Reviews.review_id, Reviews.review_date, Reviews.text
        FROM Reviews
        JOIN Biz on Reviews.biz_id = Biz.biz_id
        """ + not_str

    else:
        search_str = 'WHERE'
        j = 1
        for cname in search_terms:
            sstr = '"%' + cname + '%"'

            if j != len(search_terms):
                end_str = ' OR'
            else:
                end_str = ";"

            search_str = search_str + ' Reviews.text LIKE ' + sstr + end_str
            j += 1

        QUERY = """
        SELECT Biz.biz_id, Biz.name, Reviews.yelp_stars, Reviews.review_id, Reviews.review_date, Reviews.text
        FROM Reviews
        JOIN Biz on Reviews.biz_id = Biz.biz_id
        """ + search_str

    # print "Query: \n%s" % QUERY

    cursor = db.session.execute(QUERY)
    csearch = cursor.fetchall()
    num_results = len(csearch)
    if max_results < num_results:
        num_results = max_results
    cat_search = [random.sample(csearch, num_results)]

    return cat_search


def create_path(cat_name, class_type=None):
    """Create path where .txt files will be saved"""
    # if cat_name == 'unknown':
    #     cat_rel_path = "/data/" + cat_name + "/"
    # else:
    if not class_type:
        cat_rel_path = "/data/keywords/" + cat_name + "/"
    else:
        cat_rel_path = "/data/random_forest/"
    cat_abs_path = os.path.join(script_dir, cat_rel_path)

    return cat_abs_path


# TODO: add flag for commiting rev-cat cat_code (cat_search, cat_abs_path, db_flag)
def create_category_files(cat_search, cat_abs_path, search_terms, class_type=None):
    """exports review text as .txt files path mvp/data/training/gluten_reviews"""

    # create a .txt file of the keywords used to generate the query
    search_term_str = "|".join(search_terms)
    file_path = '.' + os.path.join(cat_abs_path, 'search_terms.txt')
    with codecs.open(file_path, 'w', 'utf-8-sig') as f:
        f.write(search_term_str)
        f.close()

    result_count = 0
    for csearch in cat_search:
        for review in csearch:
            # SELECT Biz.biz_id, Biz.name, Reviews.yelp_stars, Reviews.review_id, Reviews.review_date, Reviews.text
            biz_id = str(review[0])
            biz_name = review[1]
            stars = review[2]
            review_id = str(review[3])
            review_date = review[4]
            review_text = review[5]


            # create new text file for each review
            doc_count = '{0:04d}'.format((result_count))
            if not class_type:
                name = cat_name + str(doc_count) + '.txt'
            else:
                name = 'rf' + str(doc_count) + '.txt'
            file_path = '.' + os.path.join(cat_abs_path, name)
            print 'Creating new text file: %s' % name
            print '-'*20
            ## for debugging
            # print review_text
            # print '-'*20
            # raw_input('\nPress any key to continue.\n')

            # open and write to the file object
            # use codecs to encode as UTF-8 (handling accented characters)
            if not class_type:
                with codecs.open(file_path, 'w', 'utf-8-sig') as f:
                    f.write(review_id + '|' + biz_id + '|' + biz_name + '|' + review_date + '|' + review_text)
                    f.close()
            else:
                with codecs.open(file_path, 'w', 'utf-8-sig') as f:
                    f.write(stars + '|' + review_id + '|' + biz_id + '|' + biz_name + '|' + review_date + '|' + review_text)
                    f.close()
            result_count += 1




## Helper function for checking if input string represents an int
def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    # connect to the database
    connect_to_db(app)
    print "Connected to DB."

    print "Would you like to perform a database keyword search?"
    decision = raw_input("Y or N >> ")
    if decision.lower() == 'y':
        print 'QUERY BY KEYWORD'
        cat_name, search_terms, max_results = get_category_and_keywords()
        cat_reviews = get_category_reviews(cat_name, search_terms, max_results)

        print "Would you like to generate the category .txt files?"
        decision = raw_input("Y or N >> ")
        if decision.lower() == 'y':
            cat_abs_path = create_path(cat_name)
            print "Creating .txt files ... "
            create_category_files(cat_reviews, cat_abs_path, search_terms)
            print "File creation completed."
            print

        # option to seed query results into ReviewCategory table
        if cat_name != "unknown":
            print "Would you like to populate the results into the ReviewCategory table?"
            decision = raw_input("Y or N >> ")
            if decision.lower() == 'y':
                from server import app
                connect_to_db(app)
                print "Connected to DB."
                seed_revcat(cat_reviews, cat_name)

    else:
        print "That's cool. Maybe some other time."
        print
