from jinja2 import StrictUndefined

from flask import Flask, render_template, redirect, request, flash, session
from flask_debugtoolbar import DebugToolbarExtension

from model import YelpBiz, YelpUser, YelpReview
from model import PlatePalBiz, PlatePalUser, PlatePalRating
from model import UserList, ListEntry
from model import Category, Classification, ReviewClass, Sentiment
from model import connect_to_db, db


app = Flask(__name__)
