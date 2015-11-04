"""Utility file to seed Yelp database from Yelp Academic Dataset in data/yelp/"""

from model import YelpBiz, YelpUser, YelpReview
from model import PlatePalBiz, PlatePalUser, PlatePalReview
from model import UserList, ListEntry
from model import Category, ReviewCategory, BizSentiment
from model import connect_to_db, db
from server import app

business_filepath = 'data/yelp/yelp_academic_dataset_business.json'
review_filepath = 'data/yelp/yelp_academic_dataset_review.json'
user_filepath = 'data/yelp/yelp_academic_dataset_user.json'

## SECOND APPROACH ##

def gets_data_json(file_path):
    """
    Returns a list of JSON objects from a JSON file.

    @param file_path: the absolute path of the JSON file that contains the data
    """

    # don't process the entire review file at once, use enumerate
    # http://stackoverflow.com/questions/2081836/reading-specific-lines-only-python/2081880#2081880
    if 'review' not in file_path:
        records = [json.loads(line) for line in open(file_path)]
    else:
        records = []
        fp = open(file_path)
        for i, line in enumerate(fp):
            if i < 10000:
                record = json.loads(line)
                records.append(record)
    return records

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

    for row in open("data/yelp/yelp.biz"):
        row = row.rstrip()
        movie_entry = row.split("|")
        movie_id = movie_entry[0]
        movie_title = movie_entry[1] #TODO: need a way to remove YEAR 
        print movie_title
        # remove year frome title formated as "Title Name (YYYY)"
        # look up index of (, take title from [0:index-1]
        paren_index = movie_title.find('(')
        if paren_index != -1:
            # slice off the year and proceeding single space
            movie_title = movie_title[:(paren_index-1)]
            
        release_date = movie_entry[2]
        
        # else: # no year in title
        #     release_date = None
        
        #parse string into datetime object
        if release_date:
            rel_date_obj = datetime.strptime(release_date, '%d-%b-%Y')
        else:
            rel_date_obj = None

        # account for || before IMBd URL
        imdb_url = movie_entry[4]

        movie = Movie(movie_id=movie_id,
                      movie_title=movie_title, 
                      release_date=rel_date_obj,
                      imdb_url=imdb_url)

        db.session.add(movie)

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
    load_users()
    load_movies()
    load_ratings()