"""
Models and database functions for Hackbright Independent project.

by Patricia Decker 11/2015.
"""

# import correlation # correlation.pearson

from flask_sqlalchemy import SQLAlchemy

# This is the connection to the SQLite database; we're getting this through
# the Flask-SQLAlchemy helper library. On this, we can find the `session`
# object, where we do most of our interactions (like committing, etc.)

db = SQLAlchemy()


##############################################################################
# Model definitions

class YelpBiz(db.Model):
    """Business in Yelp Academic Dataset."""

    __tablename__ = "yelpBiz"

    #TODO: verify db.datatypes
    biz_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    address = db.Column(db.String(200), nullable=False)
    city = db.Column(db.String(64), nullable=False)
    state = db.Column(db.String(20), nullable=False)
    lat = db.Column(db.Float, nullable=False) #TODO: is there a lat/long type?
    lon = db.Column(db.Float, nullable=False)
    stars = db.Column(db.Integer, nullable=True)
    review_count = db.Column(db.Integer, nullable=True)
    photo_url = db.Column(db.String(200), nullable=True)
    is_open = db.Column(db.Boolean, nullable=False)
    url = db.Column(db.String(200), nullable=False)
    # TODO: neighborhoods
    # TODO: schools (nearby universities)

    def __repr__(self):
        return "<YelpBiz biz_id=%d name=%s>" % (self.biz_id, self.name)


class YelpUser(db.Model):
    """User in Yelp Academic Dataset."""
    
    __tablename__ = "yelpUsers"

    user_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), nullable=False)
    review_count = db.Column(db.Integer, nullable=True)  # a user might have no reviews
    average_stars = db.Column(db.Float, nullable=True)  # this is calculable from other tables
    # TODO: votes

    def __repr__(self):
        return "<YelpUser user_id=%d name=%s>" % (self.user_id, self.name)


class YelpReview(db.Model):
    """Review in Yelp Academic Dataset."""
    __tablename__ = "yelpReviews"

    # TODO: make business_id primary key??
    review_id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    biz_id = db.Column(db.Integer, db.ForeignKey('yelpBiz.biz_id'))
    user_id = db.Column(db.Integer, db.ForeignKey('yelpUsers.user_id'))
    stars = db.Column(db.Integer, nullable=False)  # integer 1-5 #TODO: restrict or check
    text = db.Column(db.Text, nullable=False)
    date = db.Column(db.Date) # TODO: verify

    def __repr__(self):
        return "<YelpReview biz_id=%d user_id=%d>" % (self.biz_id, self.user_id)


class PlatePalBiz(db.Model):
    """Business on PlatePal website. (Builds on Yelp database.)"""
    __tablename__ = "biz"

    biz_id = db.Column(db.Integer, primary_key=True) # TODO: yelp biz id?
    yelp_biz_id = db.Column(db.Integer, db.ForeignKey('yelpBiz.biz_id')) # TODO: can this be nullable???
    name = db.Column(db.String(200), nullable=False)
    address = db.Column(db.String(200), nullable=False)
    city = db.Column(db.String(64), nullable=False)
    state = db.Column(db.String(32), nullable=False)
    lat = db.Column(db.Float, nullable=False) #TODO: is there a lat/long type?
    lon = db.Column(db.Float, nullable=False)
    sen_score = db.Column(db.Float, nullable=True)
    # review_count = db.Column(db.Integer, nullable=True)
    photo_url = db.Column(db.String(200), nullable=True)
    is_open = db.Column(db.Boolean, nullable=False)
    url = db.Column(db.String(200), nullable=False)
    # TODO: neighborhoods

    def __repr__(self):
        return "<PlatePalBiz biz_id=%d name=%s>" % (self.biz_id, self.name)


class PlatePalUser(db.Model):
    """
    User of PlatePal website.
    (Builds on Yelp to allow future OAuth integration.)
    """
    __tablename__ = "users"

    user_id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(64), nullable=False)
    password = db.Column(db.String(32), nullable=False)
    fname = db.Column(db.String(32), nullable=False)
    lname = db.Column(db.String(32), nullable=False)  # is this long enough?
    bday = db.Column(db.Date, nullable=False)
    # city = db.Column(db.String(30), nullable=False)

    def __repr__(self):
        return "<PlatePalUser user_id=%d (fname lname)=%s %s>" % (self.user_id, self.fname, self.lname)


class PlatePalRating(db.Model):
    """User-generated score of a business for a specific category."""
    __tablename__ = "ratings"

    rating_id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    biz_id = db.Column(db.Integer, db.ForeignKey('biz.biz_id'))
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'))
    score = db.Column(db.Integer, nullable=False)
    cat_code = db.Column(db.Integer, db.ForeignKey('categories.cat_code'))

    def __repr__(self):
        return "<PlatePalRating rating_id=%s>" % self.rating_id


class UserList(db.Model):
    """User-generated list by category of restaurants."""
    __tablename__ = "lists"

    list_id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'))
    cat_name = db.Column(db.String(32), db.ForeignKey('categories.cat_name'))
    list_name = db.Column(db.String(64), nullable=False)

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


class Classification(db.Model):
    """
    Category determined by LinearSVC classifier.

    Classifier trained user subset of restaurant reviews in Yelp dataset."""
    __tablename__ = "classifications"

    class_id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    biz_id = db.Column(db.Integer, db.ForeignKey('biz.biz_id'))
    cat_code = db.Column(db.Integer, db.ForeignKey('categories.cat_code'))

    def __repr__(self):
        return "<Classification class_id=%s>" % self.class_id


class ReviewClass(db.Model):
    """
    Association table between reviews and classifications.

    Allows for determination of a sentiment score on an individual review,
    as a review has many categories and a category has many reviews.
    """
    __tablename__ = "revclasses"

    # TODO: using unique pairs as primary key??
    revclass_id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    review_id = db.Column(db.Integer, db.ForeignKey('yelpReviews.review_id'))
    class_id = db.Column(db.Integer, db.ForeignKey('classifications.class_id'))
    sent_score = db.Column(db.Float, nullable=False)

    def __repr__(self):
        return "<ReviewClass review_id=%s class_id=%s>" % (self.review_id, self.class_id)


class Sentiment(db.Model):
    """Calculation table for aggregate sentiment for a business-category pair."""
    __tablename__ = "sentiments"

    sent_id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    biz_id = db.Column(db.Integer, db.ForeignKey('biz.biz_id'))
    cat_code = db.Column(db.Integer, db.ForeignKey('categories.cat_code'))
    aggregate_score = db.Column(db.Float, nullable=False) # to be calculated for individual scores an updated periodically

    def __repr__(self):
        return "<Sentiment sent_id=%s>" % self.sent_id


##############################################################################
# Helper functions

def connect_to_db(app):
    """Connect the database to our Flask app."""

    # Configure to use our SQLite database
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ratings.db'
    app.config['SQLALCHEMY_ECHO'] = True
    db.app = app
    db.init_app(app)


if __name__ == "__main__":
    # As a convenience, if we run this module interactively, it will leave
    # you in a state of being able to work with the database directly.

    from server import app
    connect_to_db(app)
    print "Connected to DB."
