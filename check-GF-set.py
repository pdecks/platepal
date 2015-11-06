"""
Extract a set of reviews from the database to test
the LinearSVC category classifier.

by Patricia Decker 11/6/2015, part of Hackbright Project

As of 12:15 PM:
48 entries in JOIN
sqlite> select count(*) from reviews
   ...> JOIN biz on reviews.biz_id = biz.biz_id
   ...> WHERE reviews.text LIKE "%gluten%"
   ...> OR reviews.text LIKE "%gluten-free%"
   ...> OR reviews.text LIKE "%GF%"
   ...> OR reviews.text LIKE "%celiac%";

1836 entries in LEFT JOIN
sqlite> select count(*) from reviews
   ...> LEFT JOIN biz on reviews.biz_id = biz.biz_id
   ...> WHERE reviews.text LIKE "%gluten%"
   ...> OR reviews.text LIKE "%gluten-free%"
   ...> OR reviews.text LIKE "%GF%"
   ...> OR reviews.text LIKE "%celiac%";


"""

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import not_

import datetime

from model import YelpBiz, YelpUser, YelpReview
from model import PlatePalBiz, PlatePalUser, PlatePalReview
from model import UserList, ListEntry
from model import Category, ReviewCategory, BizSentiment
from model import connect_to_db, db
from server import app

import os
import codecs
import random
import math

# absolute dir the script is in
script_dir = os.path.dirname(__file__)

# connect to the database
connect_to_db(app)
print "Connected to DB."

# Define category name and path where .txt files will be saved
# cat_name = "unknown"
# not_name = 'gluten'
user_input = raw_input('Select set to test: "unknown" or "gluten" >> ')
cat_name = user_input.lower()
search_terms = ["gluten", "GF", "celiac", "gluten-free"]
cat_rel_path = "/data/testing/" + cat_name + "/"
cat_abs_path = os.path.join(script_dir, cat_rel_path)

# query the DB for all reviews containing a word from the category

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

    print not_str

    QUERY = "SELECT * FROM Reviews " + not_str
    print "Query: %s" % QUERY

    cursor = db.session.execute(QUERY)
    csearch = cursor.fetchall()
    cat_search = [random.sample(csearch, 2350)]

elif cat_name == 'gluten':
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

    print search_str

    QUERY = """
    SELECT Biz.name, Reviews.review_date, Reviews.text
    FROM Reviews
    JOIN Biz on Reviews.biz_id = Biz.biz_id
    """ + search_str

    print "Query: \n%s" % QUERY
    # import pdb; pdb.set_trace()
    cursor = db.session.execute(QUERY)
    csearch = cursor.fetchall()
    num_samples = len(csearch)
    # num_samples = int(math.floor(len(csearch)/4))
    cat_search = [random.sample(csearch, num_samples)]
    # cat_search = []
    # for cname in search_terms:
    #     search_str = '%' + cname + '%'
    #     csearch = PlatePalReview.query.filter(PlatePalReview.text.like(search_str)).all()
    #     cat_search.append(csearch)
else:
    print 'Exiting ...'
    quit()

# export review text as .txt files into path mvp/data/training/gluten_reviews
result_count = 0
for csearch in cat_search:
    for review in csearch:
        biz_name = review[0]
        review_date = review[1]
        review_text = review[2]
        

        # create new text file for each review
        doc_count = '{0:04d}'.format((result_count))
        name = cat_name + str(doc_count) + '.txt'
        file_path = '.' + os.path.join(cat_abs_path, name)
        print 'Creating new text file: %s' % name
        print '-'*20
        ## for debugging
        # print review_text
        # print '-'*20
        # raw_input('\nPress any key to continue.\n')

        # open and write to the file object
        # use codecs to encode as UTF-8 (handling accented characters)
        with codecs.open(file_path, 'w', 'utf-8-sig') as f:
            f.write(biz_name + '|' + review_date + '|' + review_text)
            f.close()

        result_count += 1
