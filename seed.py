"""Utility file to seed Yelp database from Yelp Academic Dataset in data/yelp/"""
import json
from model import YelpBiz, YelpUser, YelpReview
from model import PlatePalBiz, PlatePalUser, PlatePalReview
from model import UserList, ListEntry
from model import Category, ReviewCategory, BizSentiment
from model import connect_to_db, db
from server import app
from pandas import DataFrame

# filepaths to Yelp JSON
YELP_JSON_FP = 'data/yelp/yelp_academic_dataset.json'

def gets_data_frames(file_path, target_cat_list=[u'Restaurants']):
    """
    Returns pandas DataFrames containing JSON entries for users, biz, and reviews.


    file_path: the absolute path of the JSON file that contains the academic dataset
    
    target_cat_list (default u'Restaurants'): takes a list of categories for sorting
    business entries
    """

    user_records = []
    biz_records = []
    review_records = []

    fp = open(file_path)
    for line in enumerate(fp):
        # academic dataset puts lines of json in tuples...
        record = line[1].rstrip('\n')
        # convert json
        record = json.loads(record)

        if record['type'] == 'user':
            user_records.append(record)
        elif record['type'] == 'business':
            if set(target_cat_list) & set(record['categories']):
                biz_records.append(record)
        elif record['type'] == 'review':
            review_records.append(record)

    # insert all records stored in lists into respective pandas DataFrames
    udf = DataFrame(user_records)
    bdf = DataFrame(biz_records)
    rdf = DataFrame(review_records)

    return (udf, bdf, rdf)


def load_yelp_users(udf):
    """Load users from user data frame into Yelp user table."""

    print "Yelp Users"

    YelpUser.query.delete()

    for row in udf.iterrows():
        row_pd = row[1]
        user_id = row_pd['user_id']  # unicode
        name = row_pd['name']  # unicode
        review_count = row_pd['review_count']  # int
        average_stars = row_pd['average_stars']  # float

        user = YelpUser(user_id=user_id,
                      name=name,
                      review_count=review_count,
                      average_stars=average_stars
                      )

        db.session.add(user)

    db.session.commit()


def load_yelp_biz(bdf):
    """Load businesses from business data frame into Yelp biz table."""

    print "Yelp Businesses"

    YelpBiz.query.delete()

    # # business data frame
    # bdf = gets_data_frames(BUSINESS_FP)

    for row in bdf.iterrows():
        row_pd = row[1]
        biz_id = row_pd['business_id']  # unicode
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

        biz = YelpBiz(biz_id=biz_id,
                      name=name,
                      address=address,
                      city=city,
                      state=state,
                      lat=lat,
                      lng=lng,
                      stars=stars,
                      review_count=review_count,
                      is_open=is_open,
                      photo_url=photo_url,
                      )

        db.session.add(biz)

    db.session.commit()


def load_yelp_reviews(rdf):
    from datetime import datetime
    """Load reviews from yelp.review into Yelp database."""

    print "Yelp Reviews"

    YelpReview.query.delete()

    for row in rdf.iterrows():
        row_pd = row[1]
        biz_id = row_pd['business_id']  # unicode
        user_id = row_pd['user_id']  # unicode
        stars = row_pd['stars']  # integer
        text = row_pd['text']  # text
        rev_date = row_pd['date']  # date
        # format date as date object
        rev_date = datetime.strptime(rev_date, '%Y-%m-%d')

        review = YelpReview(biz_id=biz_id,
                            user_id=user_id,
                            stars=stars,
                            text=text,
                            date=rev_date
                            )

        db.session.add(review)
        db.session.commit()


if __name__ == "__main__":
    connect_to_db(app)

    # In case tables haven't been created, create them
    db.create_all()

    # Import different types of data
    udf, bdf, rdf = gets_data_frames(YELP_JSON_FP)

    load_yelp_biz(bdf)
    load_yelp_users(udf)
    load_yelp_reviews(rdf)
