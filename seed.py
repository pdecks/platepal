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
BUSINESS_FP = 'data/yelp/yelp_academic_dataset_business.json'
REVIEW_FP = 'data/yelp/yelp_academic_dataset_review.json'
USER_FP = 'data/yelp/yelp_academic_dataset_user.json'

def gets_data_frame(file_path, target_cat_list=[u'Restaurants']):
    """
    Returns a pandas DataFrame containing JSON entries.

    @param file_path: the absolute path of the JSON file that contains the data
    """

    # don't process the entire review file at once, use enumerate
    # http://stackoverflow.com/questions/2081836/reading-specific-lines-only-python/2081880#2081880
    if 'review' not in file_path:
        records = []
        if 'business' in file_path:
            for line in open(file_path):
                record = json.loads(line)
                # extract only entires that contain target categories
                if set(target_cat_list) & set(record['categories']):
                    records.append(record)
        else:
            for line in open(file_path):
                record = json.loads(line)
                records.append(record)
        # records = [json.loads(line) for line in open(file_path)]
    else:
        records = []
        fp = open(file_path)
        for i, line in enumerate(fp):
            if i < 10000:
                record = json.loads(line)
                records.append(record)

    # insert all records stored in lists to pandas DataFrame
    data_frame = DataFrame(records)

    return data_frame


def load_users():
    """Load users from yelp.user into Yelp database."""

    print "Users"

    # Delete all rows in table, so if we need to run this a second time,
    # we won't be trying to add duplicate users
    YelpUser.query.delete()

    # Read u.user file and insert data
    for row in open("data/yelp/yelp.user"):
        row = row.rstrip()
        # not all of this data is stored in database
        user_id, age, gender, occupation, zipcode = row.split("|")

        # when do we add email and pw?
        user = User(user_id=user_id,
                    age=age,
                    zipcode=zipcode)

        # We need to add to the session or it won't ever be stored
        db.session.add(user)

    # Once we're done, we should commit our work
    db.session.commit()


def load_biz():
    """Load businesses from yelp.biz into Yelp database."""

    print "Businesses"

    YelpBiz.query.delete()

    # business data frame
    bdf = gets_data_frame(BUSINESS_FP)

    for row in bdf.iterrows():
        #HELP: Integrity Errors! (lat long suspected)
        row_pd = row[1]
        # import pdb; pdb.set_trace()  # for debugging types
        biz_id = row_pd['business_id']  # unicode
        # print "This is row_pd['name']: %s" % row_pd['name']
        name = row_pd['name']  # unicode
        # print "Made it through name assignment."
        address = row_pd['full_address']  # unicode
        city = row_pd['city']  # unicode
        state = str(row_pd['state'])  # unicode
        lat = row_pd['latitude'] # float
        lng = row_pd['longitude'] # float
        stars =  row_pd['stars']  # float
        review_count = row_pd['review_count']  # int
        # is_open = row_pd['open']  # bool

        # if 'photo_url' in row_pd:
        #     photo_url = row_pd['photo_url']
        # if 'photo_url' in row_pd:
        #     yelp_url = row_pd['']

        biz = YelpBiz(biz_id=biz_id,
                      name=name,
                      address=address,
                      city=city,
                      state=state,
                      lat=lat,
                      lng=lng,
                      stars=stars,
                      review_count=review_count,
                      )

        db.session.add(biz)

    db.session.commit()


def load_reviews():
    from datetime import datetime
    """Load reviews from yelp.review into Yelp database."""

    print "Reviews"

    YelpReview.query.delete()

    for row in open("data/yelp/yelp.review"):
        row = row.rstrip()
        # rating_entry = row.split('\t')
        user_id, movie_id, score, epoch_time = row.split('\t')

        rating = Rating(user_id=user_id,
                        movie_id=movie_id,
                        score=score)

        db.session.add(rating)

    db.session.commit()


if __name__ == "__main__":
    connect_to_db(app)

    # In case tables haven't been created, create them
    db.create_all()

    # Import different types of data
    load_biz()
    # load_users()
    # load_movies()
    # load_ratings()
