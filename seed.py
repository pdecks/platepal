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
BUSINESS_FP = 'data/yelp/yelp_academic_dataset_business.json'
REVIEW_FP = 'data/yelp/yelp_academic_dataset_review.json'
USER_FP = 'data/yelp/yelp_academic_dataset_user.json'

def gets_data_frames(file_path, target_cat_list=[u'Restaurants']):
    """
    Returns pandas DataFrames containing JSON entries for users, biz, and reviews.

    @param file_path: the absolute path of the JSON file that contains the academic dataset
    """

    # don't process the entire review file at once, use enumerate
    # http://stackoverflow.com/questions/2081836/reading-specific-lines-only-python/2081880#2081880
    # if 'review' not in file_path:
    #     records = []
    #     if 'business' in file_path:
    #         for line in open(file_path):
    #             record = json.loads(line)
    #             # extract only entires that contain target categories
    #             
    #                 records.append(record)
    #     else:
    #         for line in open(file_path):
    #             record = json.loads(line)
    #             import pdb; pdb.set_trace()
    #             records.append(record)
    #     # records = [json.loads(line) for line in open(file_path)]
    # else:
    user_records = []
    biz_records = []
    review_records = []

    fp = open(file_path)
    for line in enumerate(fp):
        # academic dataset puts lines of json in tuples...
        record = line[1].rstrip('\n')
        # convert json
        record = json.loads(record)
        # import pdb; pdb.set_trace()
        if record['type'] == 'user':
            user_records.append(record)
        elif record['type'] == 'business':
            if set(target_cat_list) & set(record['categories']):
                biz_records.append(record)
        elif record['type'] == 'review':
            review_records.append(record)


    # insert all records stored in lists to pandas DataFrame
    udf = DataFrame(user_records)
    bdf = DataFrame(biz_records)
    rdf = DataFrame(review_records)

    return (udf, bdf, rdf)




# def load_users(udf):
#     """Load users from yelp.user into Yelp database."""

#     print "Users"

#     # Delete all rows in table, NOT THE TABLE ITSELF, so if we need to run this a second time,
#     # we won't be trying to add duplicate users
#     YelpUser.query.delete()

#     # Read u.user file and insert data
#     for row in open("data/yelp/yelp.user"):
#         row = row.rstrip()
#         # not all of this data is stored in database
#         user_id, age, gender, occupation, zipcode = row.split("|")

#         # when do we add email and pw?
#         user = User(user_id=user_id,
#                     age=age,
#                     zipcode=zipcode)

#         # We need to add to the session or it won't ever be stored
#         db.session.add(user)

#     # Once we're done, we should commit our work
#     db.session.commit()


def load_biz(bdf):
    """Load businesses from business data frame into Yelp database."""

    print "Businesses"

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


# def load_reviews():
#     from datetime import datetime
#     """Load reviews from yelp.review into Yelp database."""

#     print "Reviews"

#     YelpReview.query.delete()

#     for row in open("data/yelp/yelp.review"):
#         row = row.rstrip()
#         # rating_entry = row.split('\t')
#         user_id, movie_id, score, epoch_time = row.split('\t')

#         rating = Rating(user_id=user_id,
#                         movie_id=movie_id,
#                         score=score)

#         db.session.add(rating)

#     db.session.commit()


if __name__ == "__main__":
    connect_to_db(app)

    # In case tables haven't been created, create them
    db.create_all()

    # Import different types of data
    udf, bdf, rdf = gets_data_frames(YELP_JSON_FP)

    load_biz(bdf)
    # load_users()
    # load_movies()
    # load_ratings()
