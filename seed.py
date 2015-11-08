"""Utility file to seed PlatePalBiz and PlatePalReview tables from equivalent Yelp tables."""
import json

from model import CAT_CODES
from model import YelpBiz, YelpUser, YelpReview
from model import PlatePalBiz, PlatePalUser, PlatePalReview
from model import UserList, ListEntry
from model import Category, ReviewCategory, BizSentiment
from model import connect_to_db, db
from server import app
from pandas import DataFrame
from datetime import datetime

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
    from sqlalchemy.sql import func
    import random

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


def load_biz_id(n):
    """
    Populates the biz_id field for reviews in PlatePalReview

    n is the number of entries to seed.
    """

    print "... populating %d biz ids in reviews ..." % n
    print

    import random

    reviews = PlatePalReview.query.filter(PlatePalReview.biz_id.is_(None))
    reviews_n = random.sample(reviews, n)

    # lookup biz_id for each review and add to entry
    # for review in reviews_n:
    #     review_biz = PlatePalBiz.query.filter_by(yelp_biz_id=review.yelp)


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

## Helper function for checking if input string represents an int
def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    connect_to_db(app)

    # In case tables haven't been created, create them
    db.create_all()

    # Import different types of data
    # bdf, rdf = gets_data_frames(YELP_JSON_FP)

    # Seed PlatePalBiz and PlatePalReview
    # load_pp_biz(bdf)
    # load_pp_reviews(rdf)

    # Seed PlatePalReview biz_id from PlatePalBiz
    print "Would you like to fix PlatePalReview.biz_id?"
    decision = raw_input("Y or N >> ")
    if decision.lower() == 'y':
        num_to_fix = raw_input("Enter an integer value of entries to update >> ")
        num_to_offset = raw_input("Enter an integer value of entries to offset >> ")
        while not RepresentsInt(num_to_fix) or not RepresentsInt(num_to_offset):
            num_to_fix = raw_input("Enter an integer value of entries to update >> ")
            num_to_offset = raw_input("Enter an integer value of entries to offset >> ")
        fix_biz_id(int(num_to_fix), int(num_to_offset))

    else:
        pass
