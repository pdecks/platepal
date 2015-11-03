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
    business_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    address = db.Column(db.String(200), nullable=False)
    city = db.Column(db.String(64), nullable=False)
    state = db.Column(db.String(20), nullable=False)
    lat = db.Column(db.Integer, nullable=False) #TODO: is there a lat/long type?
    lon = db.Column(db.Integer, nullable=False)
    stars = db.Column(db.Integer, nullable=True)
    review_count = db.Column(db.Integer, nullable=True)
    photo_url = db.Column(db.String(200), nullable=True)
    is_open = db.Column(db.Boolean, nullable=False)
    url = db.Column(db.String(200), nullable=False)
    # TODO: neighborhoods
    # TODO: schools (nearby universities)

    def __repr__(self):
        return "<YelpBiz business_id=%d name=%s>" % (self.business_id, self.name)


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

    business_id = db.Column(db.Integer, db.ForeignKey('yelpBiz.business_id'))
    user_id = db.Column(db.Integer, db.ForeignKey('yelpUsers.user_id'))
    stars = db.Column(db.Integer, nullable=False)  # integer 1-5 #TODO: restrict or check
    # text = db.Column(db.)

    def __repr__(self):
        return "<>"


class PlatePalBiz(db.Model):
    """Business on PlatePal website. (Builds on Yelp database.)"""
    __tablename__ = "biz"

    def __repr__(self):
        return "<>"

class PlatePalUser(db.Model):
    """
    User of PlatePal website.
    (Builds on Yelp to allow future OAuth integration.)
    """
    __tablename__ = "users"

    def __repr__(self):
        return "<>"

class PlatePalRating(db.Model):
    """User-generated score of a business for a specific category."""
    __tablename__ = "ratings"

    def __repr__(self):
        return "<>"

class UserList(db.Model):
    """User-generated list by category of restaurants."""
    __tablename__ = "lists"

    def __repr__(self):
        return "<>"

class ListEntry(db.Model):
    """Restaurant entries in user-generated list, UserList."""
    __tablename__ = "entries"

    def __repr__(self):
        return "<>"

class Category(db.Model):
    """Categories for classification and targeting sentiment analysis."""
    __tablename__ = "categories"

    def __repr__(self):
        return "<>"

class Classification(db.Model):
    """
    Category determined by LinearSVC classifier.

    Classifier trained user subset of restaurant reviews in Yelp dataset."""
    __tablename__ = "classifications"

    def __repr__(self):
        return "<>"

class ReviewClassification(db.Model):
    """
    Association table between reviews and classifications.

    Allows for determination of a sentiment score on an individual review,
    as a review has many categories and a category has many reviews.
    """
    __tablename__ = "revclasses"

    def __repr__(self):
        return "<>"


class Sentiment(db.Model):
    """Calculation table for aggregate sentiment for a business-category pair."""
    __tablename__ = "sentiments"

    def __repr__(self):
        return "<>"


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
