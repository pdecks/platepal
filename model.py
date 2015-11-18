"""
Models and database functions for Hackbright Independent project.

by Patricia Decker 11/2015.
"""

# import correlation # correlation.pearson

from flask_sqlalchemy import SQLAlchemy
import datetime

# This is the connection to the SQLite database; we're getting this through
# the Flask-SQLAlchemy helper library. On this, we can find the `session`
# object, where we do most of our interactions (like committing, etc.)

db = SQLAlchemy()

CAT_CODES = {'unknown': 'unkn',
             'gluten': 'gltn',
             'allergy': 'algy',
             'paleo': 'pleo',
             'kosher': 'kshr',
             'vegan': 'vgan'
            }

##############################################################################
# Model definitions

class YelpBiz(db.Model):
    """Business in Yelp Academic Dataset."""

    __tablename__ = "yelpBiz"

    # yelp_biz_id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    biz_id = db.Column(db.Unicode(32), primary_key=True)
    name = db.Column(db.Unicode(200), nullable=False)
    address = db.Column(db.Unicode(200), nullable=False)
    city = db.Column(db.Unicode(64), nullable=False)
    state = db.Column(db.String(3), nullable=False)
    lat = db.Column(db.Float(Precision=64), nullable=False) #TODO: is there a lat/long type?
    lng = db.Column(db.Float(Precision=64), nullable=False)
    stars = db.Column(db.Float, nullable=True) # biz can have no reviews
    review_count = db.Column(db.Integer, nullable=False, default=0)
    is_open = db.Column(db.Integer, nullable=False, default=True)
    photo_url = db.Column(db.String(200), nullable=True)
    # yelp_url = db.Column(db.String(200), nullable=False)
    # V2TODO: neighborhoods
    # V2TODO: schools (nearby universities)

    def __repr__(self):
        return "<YelpBiz biz_id=%d name=%s>" % (self.biz_id, self.name)


class YelpUser(db.Model):
    """User in Yelp Academic Dataset."""
    __tablename__ = "yelpUsers"

    user_id = db.Column(db.Unicode(32), primary_key=True)
    name = db.Column(db.Unicode(64), nullable=False)
    review_count = db.Column(db.Integer, nullable=False, default=0)  # a user might have no reviews
    average_stars = db.Column(db.Float, nullable=False, default=0.0)  # this is calculable from other tables
    # V2TODO: votes

    def __repr__(self):
        return "<YelpUser user_id=%d name=%s>" % (self.user_id, self.name)


class YelpReview(db.Model):
    """Review in Yelp Academic Dataset."""

    __tablename__ = "yelpReviews"

    # TODO: make business_id primary key??
    review_id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    biz_id = db.Column(db.Unicode(32), db.ForeignKey('yelpBiz.biz_id'))
    user_id = db.Column(db.Unicode(32), db.ForeignKey('yelpUsers.user_id'))
    stars = db.Column(db.Integer, nullable=False)  # integer 1-5 #TODO: restrict or check
    text = db.Column(db.Text, nullable=False)
    date = db.Column(db.Date)

    biz = db.relationship('YelpBiz',
                          backref=db.backref('reviews', order_by=review_id))

    user = db.relationship('YelpUser',
                          backref=db.backref('reviews', order_by=review_id))

    def __repr__(self):
        return "<YelpReview biz_id=%d user_id=%d>" % (self.biz_id, self.user_id)


class PlatePalBiz(db.Model):
    """Business on PlatePal website. (Builds on Yelp database.)"""
    __tablename__ = "biz"

    biz_id = db.Column(db.Integer, autoincrement=True, primary_key=True) # TODO: yelp biz id?
    yelp_biz_id = db.Column(db.Unicode(32), db.ForeignKey('yelpBiz.biz_id')) # TODO: can this be nullable???
    name = db.Column(db.Unicode(200), nullable=False)
    address = db.Column(db.Unicode(200), nullable=False)
    city = db.Column(db.Unicode(64), nullable=False)
    state = db.Column(db.String(3), nullable=False)
    lat = db.Column(db.Float(Precision=64), nullable=False) #TODO: is there a lat/long type?
    lng = db.Column(db.Float(Precision=64), nullable=False)
    is_open = db.Column(db.Integer, nullable=False)
    photo_url = db.Column(db.String(200), nullable=True)
    # pp_url = db.Column(db.String(200), nullable=False)  # PlatePal url of biz listing
    # sen_score calculated in BizSentiments or calculable from ...
    # review_count = db.Column(db.Integer, nullable=True)
    # TODO: neighborhoods?

    def __repr__(self):
        return "<PlatePalBiz biz_id=%d name=%s>" % (self.biz_id, self.name)


class PlatePalUser(db.Model):
    """
    User of PlatePal website.
    (Builds on Yelp to allow future OAuth integration.)
    """
    # no need at moment for yelp user info

    __tablename__ = "users"

    user_id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(64), nullable=False, unique=True)
    password = db.Column(db.String(32), nullable=False)
    fname = db.Column(db.String(32), nullable=False)
    lname = db.Column(db.String(32), nullable=False)  # is this long enough?
    bday = db.Column(db.Date, nullable=False)
    # city = db.Column(db.String(30), nullable=False)
    # get around this by doing a browser request for geolocation info

    sentiments = db.relationship('ReviewCategory', secondary='reviews',
                                  backref='user')

    def __repr__(self):
        return "<PlatePalUser user_id=%d (fname lname)=%s %s>" % (self.user_id, self.fname, self.lname)


class PlatePalReview(db.Model):
    """
    Compiles Yelp Reviews, if they exist, or allows users to review a business
    directly on PlatePal.
    """
    __tablename__ = "reviews"

    review_id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    yelp_review_id = db.Column(db.Integer, db.ForeignKey('yelpReviews.review_id'))
    yelp_stars = db.Column(db.Integer, db.ForeignKey('yelpReviews.stars'))
    biz_id = db.Column(db.Integer, db.ForeignKey('biz.biz_id'))
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'))
    yelp_user_id = db.Column(db.Unicode(32), db.ForeignKey('yelpUsers.user_id'))
    cat_code = db.Column(db.Integer, db.ForeignKey('categories.cat_code'))  # this is actually a string...
    stars = db.Column(db.Integer, nullable=True)
    review_date = db.Column(db.Date, default=datetime.datetime.utcnow)
    text = db.Column(db.Text, nullable=False)


    biz = db.relationship('PlatePalBiz',
                          backref=db.backref('reviews', order_by=review_id))

    user = db.relationship('PlatePalUser',
                          backref=db.backref('reviews', order_by=review_id))

    def __repr__(self):
        return "<PlatePalReview review_id=%s date=%s>" % (self.review_id, self.review_date)


class UserList(db.Model):
    """User-generated list by category of restaurants."""
    __tablename__ = "lists"

    list_id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'))
    cat_name = db.Column(db.String(32), db.ForeignKey('categories.cat_name'))
    list_name = db.Column(db.String(64), nullable=False)

    user = db.relationship('PlatePalUser',
                           backref=db.backref('lists', order_by=list_id))


    def __repr__(self):
        return "<UserList list_id=%s user_id=%s>" % (self.list_id, self.user_id)


class ListEntry(db.Model):
    """Restaurant entries in user-generated list, UserList."""
    # TODO: is this table necessary? really storing unique pairs
    # of list_id & biz_id ...
    __tablename__ = "entries"

    entry_id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    list_id = db.Column(db.Integer, db.ForeignKey('lists.list_id'))
    biz_id = db.Column(db.Integer, db.ForeignKey('biz.biz_id'))

    user_list = db.relationship('UserList',
                                backref=db.backref('entries', order_by=biz_id))

    def __repr__(self):
        return "<ListEntry list_id=%s biz_id=%s>" % (self.list_id, self.biz_id)


class Category(db.Model):
    """Categories for classification and targeting sentiment analysis."""
    __tablename__ = "categories"

    cat_code = db.Column(db.String(4), primary_key=True)
    cat_name = db.Column(db.String(32), nullable=False)
    cat_description = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return "<Category cat_name=%s>" % self.cat_name


class ReviewCategory(db.Model):
    """
    Association table between reviews and categories.

    Allows for determination of a sentiment score on an individual review,
    as a review has many categories and a category has many reviews. Space
    to store user-generated sentiment score (feedback on machine-generated
    score).
    """

    __tablename__ = "revcats"

    # TODO: using unique pairs as primary key??
    revcat_id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    review_id = db.Column(db.Integer, db.ForeignKey('reviews.review_id'))
    biz_id = db.Column(db.Integer, db.ForeignKey('biz.biz_id'))
    cat_code = db.Column(db.Integer, db.ForeignKey('categories.cat_code')) # this is actually a string...
    sen_score = db.Column(db.Float, nullable=True)  # machine generated score
    user_sen = db.Column(db.Float, nullable=True)  # for user feedback on score

    biz = db.relationship('PlatePalBiz',
                          backref=db.backref('revcat', order_by=cat_code))

    review = db.relationship('PlatePalReview',
                             backref=db.backref('revcat', order_by=cat_code))

    def __repr__(self):
        return "<ReviewCategory revcat_id=%s>" % (self.revcat_id)


class BizSentiment(db.Model):
    """Calculation table for aggregate sentiment for a business-category pair."""
    # in postgreSQL, this would have a watcher on it (KLF)
    __tablename__ = "bizsentiments"

    bizsen_id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    biz_id = db.Column(db.Integer, db.ForeignKey('biz.biz_id'))
    cat_code = db.Column(db.Integer, db.ForeignKey('categories.cat_code'))  # this is actually a string...
    agg_sen_score = db.Column(db.Float, nullable=True)  # to be calculated for individual scores an updated periodically
    avg_cat_review = db.Column(db.Float, nullable=True)
    num_revs = db.Column(db.Integer, nullable=True)

    biz = db.relationship('PlatePalBiz',
                          backref=db.backref('sentiments', order_by=cat_code))

    def __repr__(self):
        return "<BizSentiment bizsen_id=%s>" % self.bizsen_id

    # # MVP 3a. build class/method for avg rating per cat
    # def calc_avg_rating_per_cat(self):
    #     """Calculate average stars for business by category"""

    #     # for biz_id
    #         # for a category
    #         # find all reviews in the current category
    #         # cat_code = category

    #         # take average of stars for all of those reviews
    #         # for review in list, sum = sum + review(stars)
    #         # average = sum/len(list)
    #     # update attribute
    #     # update database.


## post-MVP ##
# class ReviewSentence(db.Model):
#     """
#     Association table between review-categories and sentence-categories.

#     A review has many categories --> review-categories
#     A review-category has many sentences --> review-sentences
#     A sentence has many categories --> sentence-categories
#     """
#     __tablename__ = "revsents"

#     revsent_id = db.Column(db.Integer, autoincrement=True, primary_key=True)
#     sent_id = db.Column(db.Integer, db.ForeignKey('sentcats.sent_id'))
#     review_id = db.Column(db.Integer, db.ForeignKey('reviews.review_id')) # TODO: should this be revcat id?
#     # sent_text = db.Column(db.Text, nullable=False)
#     sen_score = db.Column(db.Float, nullable=True)

#     def __repr__(self):
#         return "<ReviewSentence sent_id=%s review_id=%s>" % (self.sent_id, self.review_id) # TODO: revcat?


class SentenceCategory(db.Model):
    """
    Association table between sentences and categories.

    A sentence has many categories <--> a category has many sentences
    """
    __tablename__ = "sentcats"

    sentcat_id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    sent_id = db.Column(db.Integer, db.ForeignKey('sentences.sent_id'))
    cat_code = db.Column(db.Integer, db.ForeignKey('categories.cat_code'))
    sen_score = db.Column(db.Integer, nullable=True)

    def __repr__(self):
        return "<SentenceCategory sent_id=%s cat_code=%s>" % (self.sent_id, self.cat_code)


class Sentence(db.Model):
    """
    Storing individual sentences of reviews, assuming sentiment analysis
    performed on a sentence-by-sentence level of granularity.
    """
    __tablename__ = "sentences"

    sent_id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    review_id = db.Column(db.Integer, db.ForeignKey('reviews.review_id'))
    sent_text = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return "<Sentence sent_id=%s review_id=%s>" % (self.sent_id, self.review_id)


class City(db.Model):
    """
    Store city information (used for geolocation of businesses)
    """
    __tablename__ = "cities"

    city_id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    city = db.Column(db.Unicode(64), nullable=False)
    state = db.Column(db.String(3), nullable=False)
    lat = db.Column(db.Float(Precision=64), nullable=False)
    lng = db.Column(db.Float(Precision=64), nullable=False)

    def __repr__(self):
        return "<City name=%s state=%s>" % (self.city, self.state)


# class NearbyCity(db.Model):
#     """ store list of nearby cities"""

#     __tablename__ = "nearbycities"

#     near_id = db.Column(db.Integer, autoincrement=True, primary_key=True)
#     city_id = db.Column(db.Integer, db.ForeignKey('cities.city_id'))
#     nearcity_id = db.Column(db.Integer, db.ForeignKey('cities.city_id'))

#     def __repr__(self):
#         return "<NearbyCity target-city=%s nearby-city=%s>" % (self.city_id, self.nearcity_id)

class CityDistance(db.Model):
    """ store distances between cities"""

    __tablename__ = "citydistances"

    distance_id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    city1_id = db.Column(db.Integer, db.ForeignKey('cities.city_id'))
    city2_id = db.Column(db.Integer, db.ForeignKey('cities.city_id'))
    miles = db.Column(db.Float(Precision=64), nullable=False) # distance in miles

    def __repr__(self):
        return "<CityDistance target-city=%s nearby-city=%s>" % (self.city1_id, self.city2_id)



##############################################################################
# Helper functions

def connect_to_db(app):
    """Connect the database to our Flask app."""

    # Configure to use our SQLite database
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///platepal.db'
    # app.config['SQLALCHEMY_BINDS'] = {'yelp': 'sqlite:///yelp.db'}
    app.config['SQLALCHEMY_ECHO'] = True
    db.app = app
    db.init_app(app)

# replaced with SQLAlchemy Bind
# def connect_to_yelp_db(app):
#     """Connect the Yelp database to our Flask app."""

#     # Configure to use our SQLite database
#     app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///yelp.db'
#     app.config['SQLALCHEMY_ECHO'] = True
#     db.app = app
#     db.init_app(app)

if __name__ == "__main__":
    # As a convenience, if we run this module interactively, it will leave
    # you in a state of being able to work with the database directly.

    from server import app

    connect_to_db(app)
    print "Connected to DB."
