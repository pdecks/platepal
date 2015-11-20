"""Utility file to seed PlatePalBiz and PlatePalReview tables from equivalent Yelp tables."""
import json
import random
import requests
import time

from model import CAT_CODES
from model import YelpBiz, YelpUser, YelpReview
from model import PlatePalBiz, PlatePalUser, PlatePalReview
from model import UserList, ListEntry
from model import Category, ReviewCategory, BizSentiment
from model import Sentence, SentenceCategory
from model import City, CityDistance
from model import connect_to_db, db

from server import app
from pandas import DataFrame
from datetime import datetime
from sqlalchemy.sql import func
from sqlalchemy import distinct
from sqlalchemy.sql import not_

from geopy.geocoders import Nominatim
from geopy.distance import vincenty

from pdclassifier import categorize_text
from pdclassifier import predict_sentiment
from pdclassifier import PennTreebankPunkt


# filepaths to Yelp JSON
YELP_JSON_FP = 'data/yelp/yelp_academic_dataset.json'

def gets_data_frames(file_path, target_cat_list=[u'Restaurants']):
    """
    Returns pandas DataFrames containing JSON entries for biz and reviews.


    file_path: the absolute path of the JSON file that contains the academic dataset

    target_cat_list (default u'Restaurants'): takes a list of categories for sorting
    business entries
    """

    biz_records = []
    review_records = []

    fp = open(file_path)
    for line in enumerate(fp):
        # academic dataset puts lines of json in tuples...
        record = line[1].rstrip('\n')
        # convert json
        record = json.loads(record)

        if record['type'] == 'business':
            if set(target_cat_list) & set(record['categories']):
                biz_records.append(record)
        elif record['type'] == 'review':
            review_records.append(record)

    # insert all records stored in lists into respective pandas DataFrames
    bdf = DataFrame(biz_records)
    rdf = DataFrame(review_records)

    return (bdf, rdf)



def load_pp_biz(bdf):
    """Load businesses from Yelp table into PlatePal table"""

    print "PlatePal Businesses"

    PlatePalBiz.query.delete()

    for row in bdf.iterrows():
        row_pd = row[1]
        yelp_biz_id = row_pd['business_id']  # unicode
        name = row_pd['name']  # unicode
        address = row_pd['full_address']  # unicode
        city = row_pd['city']  # unicode
        state = str(row_pd['state'])  # unicode
        lat = row_pd['latitude'] # float
        lng = row_pd['longitude'] # float
        stars =  row_pd['stars']  # float
        review_count = row_pd['review_count']  # int
        is_open = row_pd['open']  # bool
        if 'photo_url' in row_pd:
            photo_url = row_pd['photo_url']
        else:
            photo_url = None

        biz = PlatePalBiz(yelp_biz_id=yelp_biz_id,
                          name=name,
                          address=address,
                          city=city,
                          state=state,
                          lat=lat,
                          lng=lng,
                          is_open=is_open,
                          photo_url=photo_url,
                          )

        db.session.add(biz)

    db.session.commit()


def load_pp_reviews(rdf):
    """Load reviews from Yelp table into PlatePal table"""

    print "PlatePal Reviews"

    PlatePalReview.query.delete()

    # update for reviews in businesses only ...
    for row in rdf.iterrows():
        row_pd = row[1]
        yelp_biz_id = row_pd['business_id']  # unicode

        # check if business is in YelpBiz Table
        # if not, skip review entry

        # import pdb; pdb.set_trace()
        check_biz = YelpBiz.query.filter(YelpBiz.biz_id==yelp_biz_id).first()
        # check_pp_biz = PlatePalBiz.query.filter_by(yelp_biz_id=biz_id).first()
        if not check_biz:
            continue
        # else, add review to database
        else:
            yelp_user_id = row_pd['user_id']  # unicode
            # yelp_review = YelpReview.query.filter((YelpReview.biz_id==biz_id) & (YelpReview.user_id==yelp_user_id)).one()
            # yelp_review_id = yelp_review.review_id
            yelp_stars = row_pd['stars']  # integer
            text = row_pd['text']  # text
            rev_date = row_pd['date']  # date
            # format date as date object
            rev_date = datetime.strptime(rev_date, '%Y-%m-%d')

            # will have to insert review ID later
            review = PlatePalReview(yelp_review_id=None,
                                    yelp_stars=yelp_stars,
                                    yelp_biz_id=yelp_biz_id,
                                    user_id=None,
                                    yelp_user_id=yelp_user_id,
                                    cat_code=None,
                                    stars=None,
                                    review_date=rev_date,
                                    text=text
                                     )

        db.session.add(review)
        db.session.commit()


def fix_biz_id(num_to_fix, num_to_offset):
    """
    Moves biz_id entry to yelp_biz_id field for all reviews

    num_to_fix is the number of entries to fix
    """

    # select only reviews where the 22-character yelp_biz_id is in the biz_id field
    reviews = PlatePalReview.query.filter(func.length(PlatePalReview.biz_id)==22).limit(num_to_fix).offset(num_to_offset)
    # reviews_n = random.sample(reviews, num_to_fix)

    for review in reviews:
        # import pdb; pdb.set_trace()
        yelp_biz_id = review.biz_id
        review_biz = PlatePalBiz.query.filter_by(yelp_biz_id=yelp_biz_id).first()
        review.biz_id = review_biz.biz_id
        db.session.commit()
    return


# TODO This looks like it can be deleted ...
# def load_biz_id(n):
#     """
#     Populates the biz_id field for reviews in PlatePalReview

#     n is the number of entries to seed.
#     """

#     print "... populating %d biz ids in reviews ..." % n
#     print

#     import random

#     reviews = PlatePalReview.query.filter(PlatePalReview.biz_id.is_(None))
#     reviews_n = random.sample(reviews, n)

#     # lookup biz_id for each review and add to entry
#     # for review in reviews_n:
#     #     review_biz = PlatePalBiz.query.filter_by(yelp_biz_id=review.yelp)


def seed_revcat(cat_search, category):
    """Used by keywordsearch.py to populate RevCat table with cat_reviews"""

    # lookup cat_code in CAT_CODES dict imported from model
    cat_code = CAT_CODES[category]
    # iterate over entries in cat_reviews
    for csearch in cat_search:
        for review in csearch:
            biz_id = review[0]
            biz_name = review[1]
            review_id = review[2]
            review_date = review[3]
            review_text = review[4]

            # check whether review id / cat_code pair already in db
            revcat = ReviewCategory.query.filter(ReviewCategory.review_id == review_id, ReviewCategory.cat_code == cat_code).first()

            # if not exists, add
            if revcat is None:
                revcat = ReviewCategory(review_id=review_id,
                                        biz_id=biz_id,
                                        cat_code=cat_code
                                        )
                db.session.add(revcat)
                db.session.commit()
    return

def seed_keyword_revcat(search_term, cat_code):
    """Add more revcats by using like '%vegan%'

    Tested on all reviews containing 'vegan' where reviews.biz_id=148, then
    reran for all reviews containing 'vegan' where reviews.biz_id != 148
    """
    #  sqlite> select reviews.review_id, revcats.revcat_id, sentences.sent_id from reviews
    # ...> LEFT JOIN revcats ON revcats.review_id = reviews.review_id
    # ...> LEFT JOIN sentences on sentences.review_id = reviews.review_id
    # ...> WHERE reviews.biz_id = 148 and reviews.text like '%vegan%';

    # sqlite> select count(*) from reviews where reviews.biz_id != 148 and reviews.text like '%vegan%';
    # count(*)
    # 3946

    # query db for all reviews containing the word 'vegan'
    # search_term = 'vegan'
    reviews = db.session.query(PlatePalReview, ReviewCategory, Sentence, SentenceCategory)
    reviews_joined = reviews.outerjoin(ReviewCategory).outerjoin(Sentence).outerjoin(SentenceCategory)
    keyword_reviews = reviews_joined.filter(PlatePalReview.text.like(('%'+search_term+'%')))
    # vegan_reviews = reviews_joined.filter(PlatePalReview.biz_id!=148, PlatePalReview.review_id!=7617, PlatePalReview.text.like(('%'+search_term+'%')))

    # instantiate preprocessor for splitting text into sentences
    preprocessor = PennTreebankPunkt(use_flag="sentences")

    for group in keyword_reviews:
        review = group[0]
        revcats = group[1]
        sentences = group[2]
        sentcats = group[3]

        # check if review has revcats
        if not revcats:
            # get sentiment score of review
            sen_score = get_sentiment(review.text)
            # add review to revcat 'vgan'
            revcat = ReviewCategory(review_id=review.review_id,
                                    biz_id=review.biz_id,
                                    cat_code=cat_code,
                                    sen_score=sen_score,
                                    )
            db.session.add(revcat)
            db.session.commit()
        else: # there are revcats
            pass

        # check if review has sentences
        if not sentences:
            # tokenize into sentences and add to sentences
            sentence_list = preprocessor(review.text)
            # add sentence to Sentences table
            for sentence in sentence_list:
                sent = Sentence(review_id=review.review_id,
                                sent_text=sentence
                                )
                db.session.add(sent)
                db.session.commit()
                # add sentences containing search_term to sentcats
                if search_term in sentence:
                    sent_id = db.session.query(Sentence.sent_id).filter(Sentence.sent_text==sentence, Sentence.review_id==review.review_id).all()
                    if sent_id:
                        for sid in sent_id:
                            # import pdb; pdb.set_trace()
                            # get sentiment score of sentence
                            sen_score = get_sentiment(sentence)
                            sentcat = SentenceCategory(sent_id=sid[0],
                                                       cat_code=cat_code,
                                                       sen_score=sen_score)
                            db.session.add(sentcat)
                            db.session.commit()
                else:
                    pass
        else: #there are sentences, so check if sentences containing search_term have sentcats
            if not sentcats:
                # check if more than one sentence
                if isinstance(type(sentences), list):
                    for sentence in sentences:
                        if search_term in sentence.text:

                            sent_id = db.session.query(Sentence.sent_id).filter(Sentence.sent_text==sentence, Sentence.review_id==review.review_id).all()
                            if sent_id:
                                for sid in sent_id:
                                    # get sentiment score of sentence TODO fix
                                    sen_score = get_sentiment(sentence.sent_text)
                                    sentcat = SentenceCategory(sent_id=sid[0],
                                                               cat_code=cat_code,
                                                               sen_score=sen_score)
                                    db.session.add(sentcat)
                                    db.session.commit()
                else: # single sentence in sentences
                    sentence = sentences
                    if search_term in sentence.sent_text:
                        # get sentiment score of sentence
                        sen_score = get_sentiment(sentence.sent_text)
                        sentcat = SentenceCategory(sent_id=sentence.sent_id,
                                                   cat_code=cat_code,
                                                   sen_score=sen_score)
                        db.session.add(sentcat)
                        db.session.commit()
            else: # there are sencats ... make sure cat_code matches sencat.cat_code
                # check if sentiment score exists
                if isinstance(type(sentcats), list):
                    for sentcat in sentcats:
                        if sentcat.cat_code == cat_code:
                            sentence_text = db.session.query(Sentence.sent_text).filter(Sentence.sent_id==sentcat.sent_id).one()
                            if search_term in sentence_text:
                                if not sentcat.sen_score:
                                    # get sentiment score of sentence
                                    sen_score = get_sentiment(sentence_text)
                                    sentcat.sen_score = sen_score
                                    db.session.add(sentcat)
                                    db.session.commit()
                                elif sentcat.sen_score == 0:
                                    # get sentiment score of sentence
                                    sen_score = get_sentiment(sentence_text)
                                    sentcat.sen_score = sen_score
                                    db.session.add(sentcat)
                                    db.session.commit()
                                else: # there is a non-zero sentiment score
                                    print "sentiment score exists for sentcat %d", sentcat.sentcat_id
                        else: # sentcat.cat_code != cat_code
                            pass
                else: #single sentcat
                    sentcat = sentcats
                    if sentcat.cat_code == cat_code:
                        sentence_text = db.session.query(Sentence.sent_text).filter(Sentence.sent_id==sentcat.sent_id).one()
                        if search_term in sentence_text:
                            if not sentcat.sen_score:
                                # get sentiment score of sentence
                                sen_score = get_sentiment(sentence.sent_text)
                                sentcat.sen_score = sen_score
                                db.session.add(sentcat)
                                db.session.commit()
                            elif sentcat.sen_score == 0:
                                # get sentiment score of sentence
                                sen_score = get_sentiment(sentence.sent_text)
                                sentcat.sen_score = sen_score
                                db.session.add(sentcat)
                                db.session.commit()
                            else: # there is a non-zero sentiment score
                                print "sentiment score exists for sentcat %d", sentcat.sentcat_id
                    else: #sentcat.cat_code != cat_code
                        pass
    return


def update_revcat_sen_score(cat='gltn'):
    """Update RevCat table with sen_scores"""
    # select all revcat entries where cat_code == category and return the review text
    results = db.session.query(ReviewCategory.revcat_id, PlatePalReview.text).join(PlatePalReview)
    results_by_cat = results.filter(ReviewCategory.cat_code==cat).all()
    # for the list of revcats / review text
    for result in results_by_cat:
        revcat_id = result[0]
        text = result[1]
        # note: predict_sentiment components revived in function
        sentiment_score = predict_sentiment([text])
        # store prediction_list[0][0][2] (decision_function score) as sen_score
        sen_score = sentiment_score[0][2]

        # update entry in db --> get entire entry from revcat by revcat_id
        revcat = db.session.query(ReviewCategory).filter(ReviewCategory.revcat_id==revcat_id).one()
        revcat.sen_score = sen_score
        # print "this is revcat.revcat_id", revcat.revcat_id
        # print "this is revcat.sen_score", revcat.sen_score
        db.session.add(revcat)

    db.session.commit()
    print "... database updated!"

    return

def replace_revcat_sen_score(cat='gltn'):
    """Replace sen_scores in RevCat table with text-processing API scores"""
    url = "http://text-processing.com/api/sentiment/"
    # select all revcat entries where cat_code == category and return the review text
    results = db.session.query(ReviewCategory.revcat_id, PlatePalReview.text).join(PlatePalReview)
    results_by_cat = results.filter(ReviewCategory.cat_code==cat).all()
    # for the list of revcats / review text
    for result in results_by_cat:
        revcat_id = result[0]
        text = result[1]
        # check that text does not exceed API's character limit
        if len(text) < 80000:
            # query text-processing API for sentiment score
            payload = {'text': text}

            # make API call
            r = requests.post(url, data=payload)

            # load JSON from API call
            result = json.loads(r.text)

            # pull sentiment score
            sen_score = result['probability']['pos']

            time.sleep(random.randint(0,10))
        # update entry in db --> get entire entry from revcat by revcat_id
        revcat = db.session.query(ReviewCategory).filter(ReviewCategory.revcat_id==revcat_id).one()
        if revcat:
            revcat.sen_score = sen_score
            db.session.add(revcat)
            db.session.commit()

    print "... database updated!"

    return


def seed_sentences():
    """
    For reviews in RevCats, split reviews into sentences and store
    sentences in Sentences table.
    """
    decision = raw_input("Are you sure you want to seed SENTENCES table? Y or N")
    if decision.lower() == 'y':
        # instantiate preprocessor imported from pdclassifier.py
        preprocessor = PennTreebankPunkt(use_flag="sentences")
        # query db for reviews in revcats
        results = db.session.query(PlatePalReview.review_id, PlatePalReview.text).join(ReviewCategory).all()

        # for each review...
        for review in results:
            # split reviews into sentences
            sentence_list = preprocessor(review.text)
            # add sentence to Sentences table
            for sentence in sentence_list:
                sent = Sentence(review_id=review.review_id,
                                sent_text=sentence
                                )
                db.session.add(sent)
            db.session.commit()
    else:
        print "Phew! That was close."
    return


def seed_sentcats():
    """
    For sentences in Sentences, categorize using multilabel classifier
    Add results to SentCats -- Initial seeding version
    """
    # select all sentences from Sentences table
    results = db.session.query(Sentence).offset(0).all()
    # for each sentence, categorize with classifier
    for sentence in results:
        sent_id = sentence.sent_id
        text = sentence.sent_text
        predictions = categorize_text(text)
        for cat in predictions:
            # for 'gltn', perform sentiment analysis
            if cat == 'gltn':
                # note: predict_sentiment components revived in function
                sentiment_score = predict_sentiment([text])
                # store prediction_list[0][0][2] (decision_function score) as sen_score
                sen_score = sentiment_score[0][2]
                # query db to check for entry
                sentcat = SentenceCategory.query.filter(SentenceCategory.sent_id==sent_id).first()
                if not sentcat:
                    sentcat = SentenceCategory(sent_id=sent_id,
                                               cat_code='gltn',
                                               sen_score=sen_score)
                else:
                    sentcat.sen_score=sen_score
            else:
                # query db to check for entry
                sentcat = SentenceCategory.query.filter(SentenceCategory.sent_id==sent_id).first()
                if not sentcat:
                # TODO: will have to perform sentiment analysis and update later
                    sentcat = SentenceCategory(sent_id=sent_id,
                                               cat_code=cat
                                               )
                else:
                    pass
            db.session.add(sentcat)
        db.session.commit()
    return


def update_sentcat_score(cat_code, search_term):
    """Replace hand-built sentiment score with text-processing API score"""
    # checking progress of update_sentcat_score('vgan', 'vegan')
    #  sqlite> select sentences.sent_text, sentcats.sentcat_id, sentcats.sen_score from sentences
    # ...> left join sentcats on sentcats.sent_id = sentences.sent_id
    # ...> where sentcats.cat_code = 'vgan'
    # ...> limit 10;

    url = "http://text-processing.com/api/sentiment/"

    updated_cat_codes = ['gltn', 'algy']
    # get all sentences containing search term
    sentences = Sentence.query.filter(Sentence.sent_text.like('%'+search_term+'%')).all()

    # get inverse sentences and set sen_score = 0
    # ! check this ! sentences = SentenceCategory.query.outerjoin(Sentence).filter((not_(Sentence.sent_text.like('%gluten%'))) | (not_(Sentence.sent_text.like('%celiac%')))).all()
    for sentence in sentences:
        # query text-processing API for sentiment score
        doc = sentence.sent_text
        payload = {'text': doc}

        # make API call
        r = requests.post(url, data=payload)

        # load JSON from API call
        result = json.loads(r.text)

        # pull sentiment score
        sen_score = result['probability']['pos']

        # check if sentence is in sentcat
        result = SentenceCategory.query.filter(SentenceCategory.sent_id==sentence.sent_id).one()
        if result:
            # don't update gltn reviews again
            if result.cat_code not in updated_cat_codes:
                # update sen_score
                result.sen_score = sen_score
        else:
            # add sentence to sentcat
            sentcat = SentenceCategory(sent_id=sentence.sent_id,
                                       cat_code=cat_code,
                                       sen_score=sen_score)
        # sentence.sen_score = 0
        db.session.commit()

        # wait 5 seconds before making the next call
        time.sleep(random.randint(0,10))
    return


# MVP 3a. build class/method for avg rating per cat
# this should only be applied to the businesses that are in revcats, as the other
# businesses are classified as "unknown" and therefore don't have a category
# or their scores for the category would be unknown
def calc_avg_rating_per_cat():
    """Calculate average stars for business by category"""

    # 1. find the businesses having more than one revcat (multiple reviews for a business)
    # SELECT biz_id, COUNT(cat_code) as num_revcats FROM revcats GROUP BY biz_id HAVING COUNT(cat_code) > 1 ORDER BY COUNT(cat_code) DESC;

    # 2. find the businesses with more than one cat_code (multiple categories within multiple reviews)
    # SELECT biz_id, COUNT(DISTINCT cat_code) as num_cats FROM revcats GROUP BY biz_id HAVING COUNT(cat_code) > 1 ORDER BY COUNT(DISTINCT cat_code) DESC;
    # --> SELECT biz_id as num_cats FROM revcats GROUP BY biz_id HAVING COUNT(cat_code) > 1 ORDER BY COUNT(DISTINCT cat_code) DESC;
    # multiple_cat_biz = db.session.query(ReviewCategory.biz_id).group_by(ReviewCategory.biz_id).having(func.count(ReviewCategory.cat_code)>1).order_by(func.count(distinct(ReviewCategory.cat_code))).all()
    # 3. for each of these biz_ids, select the cat codes

    # query ReviewCategory for unique biz_ids
        # revcat_biz = db.session.query(distinct(ReviewCategory.biz_id)).all()
        # revcat_biz = db.session.query(distinct(ReviewCategory.biz_id), ReviewCategory.cat_code).order_by(ReviewCategory.cat_code).all()

    revcats = db.session.query(ReviewCategory.review_id, ReviewCategory.biz_id, ReviewCategory.cat_code).order_by(ReviewCategory.biz_id).all()
    unique_biz = set([revcat[1] for revcat in revcats])
    #for each biz_id with more than one review
    for biz in unique_biz:
        biz_id = biz

        # find distinct categories for biz in revcat
        cats = set([revcat[2] for revcat in revcats if revcat[1] == biz_id])
        # bizcats = db.session.query(distinct(ReviewCategory.cat_code)).filter(ReviewCategory.biz_id==biz_id).all()

        # for a category
        for cat in cats:
            cat_code = cat
            # find all reviews in the current category
            revs = [revcat[0] for revcat in revcats if (revcat[1] == biz_id and revcat[2] == cat)]
            # revs = db.session.query(ReviewCategory.review_id).filter(ReviewCategory.biz_id==biz_id, ReviewCategory.cat_code==cat_code).all()

            # take average of stars for all of those reviews
            sum_stars_by_cat = 0
            # for each review in category, get num of stars from reviews table
            num_revs = len(revs)
            for rev in revs:
                review_id = rev
                stars = db.session.query(PlatePalReview.yelp_stars).filter(PlatePalReview.review_id==review_id).first()
                # update sum
                sum_stars_by_cat += stars[0]
                average_stars_by_cat = (sum_stars_by_cat / num_revs) / 1.0

            # update attribute
            bizsen_cat = BizSentiment(biz_id=biz_id,
                                  cat_code=cat_code,
                                  avg_cat_review=average_stars_by_cat,
                                  num_revs=num_revs)

            # update db
            db.session.add(bizsen_cat)
            db.session.commit()
    return


def calc_agg_sen_per_cat():
    """Calculate aggregate sentiment score for business by category"""
    # 1. List of PLATEPALBIZ: find the businesses with at least one revcat in cat_code = 'gltn'
    # 2. List of REVCATS for Biz: for that business, find all of its reviews in the category
    # 1 + 2 --> query for revcat in category, for each revcat, find .biz --> take set of these biz
    #           ... then for each unique use backref biz.revcat --> this is the list of revcats
    # ALTERNATIVELY:, assuming bizsents already seeded with avg_rating by stars
    #           ... bizsents = BizSentiment.query.filter(BizSentiment.cat_code==cat).all()
    # select count(biz_id) from bizsentiments where cat_code = 'gltn';
    # 3. Average sentiment scores (sen_score) for all of those reviews and store as agg_sen_score in BIZSENTIMENTS
    # 4. Update database

    # query ReviewCategory for unique biz_ids (215 for 'gltn' 11/18/2015)
    cat = 'gltn'
    revcat_biz = db.session.query(ReviewCategory).filter(ReviewCategory.cat_code==cat).group_by(ReviewCategory.biz_id).all()
    for revcat in revcat_biz:
        biz = revcat.biz

        biz_revcats = biz.revcat

        agg_sen_score = 0.0
        total_sen_score = 0.0
        num_scores = 0
        # calculate average
        for entry in biz_revcats:
            if entry.cat_code == 'gltn':
                num_scores += 1
                total_sen_score += entry.sen_score

        agg_sen_score = (total_sen_score / num_scores) / 1.0
        # store average in db BizSentiments
        # query and update...
        print "\nUpdating bizsentiments table for biz_id=%d ...\n" % biz.biz_id
        bizsent = BizSentiment.query.filter(BizSentiment.biz_id==biz.biz_id, BizSentiment.cat_code==cat).first()
        bizsent.agg_sen_score = agg_sen_score
        print "\n... database updated.\n"

        db.session.commit()
    return


def seed_cities():
    """Add all cities in Biz table to Cities table"""
    # should be 95 cities
    # select city, state from biz group by state, city
    # group by state, city
    all_cities = db.session.query(PlatePalBiz.city, PlatePalBiz.state).filter(PlatePalBiz.city!=u"blacksburg", PlatePalBiz.city!=u'Carrboro Saxapahaw Chapel Hill Durham', PlatePalBiz.city!=u'Greenbelt ')
    cities = all_cities.group_by(PlatePalBiz.state).group_by(PlatePalBiz.city).all()

    # calculate lat/lng for each city
    geolocator = Nominatim()
    for city in cities:
        location = geolocator.geocode(city[0] + " " + city[1])
        print city
        print "Lat: {}, Lng: {}".format(location.latitude, location.longitude)
        new_city = City(city=city[0],
                        state=city[1],
                        lat=location.latitude,
                        lng=location.longitude)
        db.session.add(new_city)
    db.session.commit()


def seed_city_distance():
    """populate city distances table"""
    # should be 95 cities
    # select city, state from biz group by state, city
    # group by state, city
    cities = db.session.query(City)

    # find nearby cities (<50 miles)
    for city in cities:
        city1 = (city.lat, city.lng)
        for other_city in cities:
            if other_city != city:
                city2 = (other_city.lat, other_city.lng)
                # evaluate distance
                miles = vincenty(city1, city2).miles

                new_city_distance = CityDistance(city1_id=city.city_id,
                                                 city2_id=other_city.city_id,
                                                 miles=miles)
                db.session.add(new_city_distance)
    db.session.commit()


def update_ppreview_cat():
    """
    Correct review.cat_code in PlatePalReview

    Incorrectly defined in model.py when initially seeded tables.
    Because a review can have multiple categories, RevCats was used to
    store the categories for a review.

    Review.cat_code can store a string with the cat codes, e.g.
    'gltnvganpleo', which can be parsed 4 characters at a time to break
    off the cat codes.
    """
    pass  # TODO


## Helper function for checking if input string represents an int
def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

## helper function for calling text-processing API
def get_sentiment(text):
    """Call text-processing API and get sentiment of text
    note API limits 80,000 characters per text, 1000 calls
    per IP address
    """
    # check that text does not exceed API's character limit
    url = "http://text-processing.com/api/sentiment/"
    if len(text) < 80000:
        # query text-processing API for sentiment score
        payload = {'text': text}

        # make API call
        r = requests.post(url, data=payload)

        # load JSON from API call
        result = json.loads(r.text)

        # pull sentiment score
        sen_score = result['probability']['pos']

        time.sleep(random.randint(0,5))
    return sen_score

if __name__ == "__main__":
    connect_to_db(app)

    # In case tables haven't been created, create them
    db.create_all()

    # Import different types of data
    # bdf, rdf = gets_data_frames(YELP_JSON_FP)

    # Seed PlatePalBiz and PlatePalReview
    # load_pp_biz(bdf)
    # load_pp_reviews(rdf)

    # FOR FIXING PlatePalBiz BIZ IDs (fixed as of 11/8/2015)
    # Seed PlatePalReview biz_id from PlatePalBiz
    # print "Would you like to fix PlatePalReview.biz_id?"
    # decision = raw_input("Y or N >> ")
    # if decision.lower() == 'y':
    #     num_to_fix = raw_input("Enter an integer value of entries to update >> ")
    #     num_to_offset = raw_input("Enter an integer value of entries to offset >> ")
    #     while not RepresentsInt(num_to_fix) or not RepresentsInt(num_to_offset):
    #         num_to_fix = raw_input("Enter an integer value of entries to update >> ")
    #         num_to_offset = raw_input("Enter an integer value of entries to offset >> ")
    #     fix_biz_id(int(num_to_fix), int(num_to_offset))

    # else:
    #     pass

    # print "Would you like to seed BizSentiment by category?"
    # decision = raw_input("Y or N >> ")
    # if decision.lower() == 'y':
    #     calc_avg_rating_per_cat()

    # print "Would you like to seed Cities?"
    # decision = raw_input("Y or N >> ")
    # if decision.lower() == 'y':
    #     seed_cities()

    # print "Would you like to seed NearbyCities?"
    # decision = raw_input("Y or N >> ")
    # if decision.lower() == 'y':
    #     seed_city_distance()
