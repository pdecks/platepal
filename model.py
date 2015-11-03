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
    __tablename__ = "yelpBusinesses"

    pass


class YelpUser(db.Model):
    """User in Yelp Academic Dataset."""
    __tablename__ = "yelpUsers"

    pass


class YelpReview(db.Model):
    """Review in Yelp Academic Dataset."""
    __tablename__ = "yelpReviews"

    pass


class PlatePalBiz(db.Model):
    """Business on PlatePal website. (Builds on Yelp database.)"""
    __tablename__ = "businesses"

    pass

class PlatePalUser(db.Model):
    """
    User of PlatePal website.
    (Builds on Yelp to allow future OAuth integration.)
    """
    __tablename__ = "users"

    pass

class PlatePalRating(db.Model):
    """User-generated score of a business for a specific category."""
    __tablename__ = "ratings"

    pass

class UserList(db.Model):
    """User-generated list by category of restaurants."""
    __tablename__ = "lists"

    pass

class ListEntry(db.Model):
    """Restaurant entries in user-generated list, UserList."""
    __tablename__ = "entries"

    pass

class Category(db.Model):
    """Categories for classification and targeting sentiment analysis."""
    __tablename__ = "categories"

    pass

class Classification(db.Model):
    """
    Category determined by LinearSVC classifier.

    Classifier trained user subset of restaurant reviews in Yelp dataset."""
    __tablename__ = "classifications"

    pass

class ReviewClassification(db.Model):
    """
    Association table between reviews and classifications.

    Allows for determination of a sentiment score on an individual review,
    as a review has many categories and a category has many reviews.
    """
    __tablename__ = "revclasses"

    pass

class Sentiment(db.Model):
    """Calculation table for aggregate sentiment for a business-category pair."""
    __tablename__ = "sentiments"

    pass


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
