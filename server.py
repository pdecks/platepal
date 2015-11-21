from jinja2 import StrictUndefined

from flask import Flask, render_template, redirect, request, flash, session, jsonify
from flask_debugtoolbar import DebugToolbarExtension

from model import YelpBiz, YelpUser, YelpReview
from model import PlatePalBiz, PlatePalUser, PlatePalReview
from model import UserList, ListEntry
from model import Category, ReviewCategory, BizSentiment
from model import City, CityDistance
from model import connect_to_db, db

from sqlalchemy import distinct
from model import CAT_CODES
from statecodes import STATE_CODES

import re
import os
import requests
import json

# from geopy.geocoders import Nominatim

CAT_CODES_ID = ['gltn', 'vgan', 'kshr', 'algy', 'pleo', 'unkn']
CAT_NAMES_ID = ['Gluten-Free', 'Vegan', 'Kosher', 'Allergies', 'Paleo', 'Feeling Lucky']
CAT_DICT = {CAT_CODES_ID[n]: CAT_NAMES_ID[n] for n in range(len(CAT_CODES_ID))}
CAT_LISTS = [[CAT_CODES_ID[n], CAT_NAMES_ID[n]] for n in range(len(CAT_CODES_ID))]
google_maps_key = os.environ['GOOGLE_MAPS_API_KEY']

app = Flask(__name__)

# Required to use Flask sessions and the debug toolbar
app.secret_key = "ABC"

# Normally, if you use an undefined variable in Jinja2, it fails silently.
# This is horrible. Fix this so that, instead, it raises an error.
app.jinja_env.undefined = StrictUndefined

@app.route('/')
def index():
    """Homepage."""
    city = 'Palo Alto'
    state = 'CA'
    x_miles = 10
    nearby_cities = find_nearby_cities(city, state, x_miles)
    return redirect('/CA/Palo%20Alto/city.html')


@app.route('/analytics')
def lab():
    """Analytics page housing sentiment analysis info and D3"""
    return render_template('/analytics.html')


@app.route('/<state>/state.html')
def display_all_biz_in_state(state):
    """State landing page."""

    # query for cities in state
    #  select distinct(Biz.city) from Biz
    # ...> Join Reviews on Biz.biz_id = Reviews.biz_id
    # ...> where Biz.state = state;
    state_all_cities = db.session.query(PlatePalBiz.city).join(PlatePalReview).filter(PlatePalBiz.state==state)
    state_cities = state_all_cities.group_by(PlatePalBiz.city).all()
    print "this is state_cities", state_cities
    return render_template('state.html', google_maps_key=google_maps_key, cat_list=CAT_LISTS, state=state, state_name=STATE_CODES[state], state_cities=state_cities)


@app.route('/<state>/state.json')
def get_all_biz_in_state(state):
    """ """
    # select Biz.biz_id, Biz.name, Biz.city from reviews
    # join Biz on biz.biz_id = reviews.biz_id
    # where reviews.cat_code = 'gltn' and biz.city='Palo Alto' limit 100;
    state_biz = db.session.query(PlatePalBiz).outerjoin(ReviewCategory).filter(PlatePalBiz.state==state)
    state_biz_GF = state_biz.filter(ReviewCategory.cat_code=='gltn').all()
    data_list_of_dicts = {}
    gltn_list = []
    for biz in state_biz_GF:
        biz_dict = {'biz_id': biz.biz_id,
                    'name': biz.name,
                    'lat': biz.lat,
                    'lng': biz.lng,
                    }
        gltn_list.append(biz_dict)
        data_list_of_dicts['gltn'] = gltn_list

    return jsonify(data_list_of_dicts)


@app.route('/<state>/<city>/city.html')
def city_in_state_page(state, city):
    """City Homepage"""

    # get nearby cities list
    nearby_miles = 50 # find cities within 50 miles of current city
    nearby_cities = find_nearby_cities(city, state, nearby_miles)

    return render_template('city.html', google_maps_key=google_maps_key, cat_list=CAT_LISTS, state=state, state_name=STATE_CODES[state], city=city, nearby_cities=nearby_cities)


# select distinct biz.biz_id, biz.name from biz join revcats on biz.biz_id = revcats.biz_id where revcats.cat_code = 'gltn' and biz.city='Palo Alto';
@app.route('/<state>/<city>/city.json')
def display_all_reviews_in_city(state, city):

    categories = dict(CAT_CODES)
    # del categories['unknown']

    data_list_of_dicts = {}
    # query database for top 5 businesses for each category
    for cat_name in categories:
        print "this is cat_name", cat_name
        cat_code = categories[cat_name]

        # select distinct Biz.biz_id, Biz.name, Biz.city from Biz
        # join Bizsentiments on biz.biz_id = bizsentiments.biz_id
        # where BizSentiments.cat_code = 'gltn' and biz.state='CA' and biz.city='Palo Alto';
        state_biz = db.session.query(PlatePalBiz, BizSentiment).join(BizSentiment).filter(PlatePalBiz.state==state)
        city_biz = state_biz.filter(PlatePalBiz.city==city)

        if cat_code != 'unkn':
            city_biz_cat = city_biz.filter(BizSentiment.cat_code==cat_code).all()
        else:
            city_biz_cat = city_biz.group_by(PlatePalBiz.biz_id).all()

        cat_list = []
        for biz, bizsen in city_biz_cat:

            biz_dict = {'biz_id': biz.biz_id,
                        'name': biz.name,
                        'address': biz.address,
                        'city': biz.city,
                        'state': biz.state,
                        'lat': biz.lat,
                        'lng': biz.lng,
                        'is_open': biz.is_open,
                        'photo_url': biz.photo_url,
                        'avg_cat_review': bizsen.avg_cat_review,
                        'agg_sen_score': bizsen.agg_sen_score,
                        'num_revs': bizsen.num_revs
                        }
            # if biz_dict not in cat_list:
            cat_list.append(biz_dict)

        data_list_of_dicts[cat_code] = cat_list

    return jsonify(data_list_of_dicts)


@app.route('/popular-biz.json')
def popular_biz_data():
    """
    Return data about popular businesses

    JSON = {'gltn': [{'biz_id': biz_id, 'avg_cat_review': avg_cat_review, 'lat': lat, 'lng': lng}, {}, {}],
            'vgan': [{}, {}, {}],
             ...
            }
    """
    print "in popular-biz.json"

    # remove 'unknown' category from list of categories
    categories = dict(CAT_CODES)
    del categories['unknown']

    data_list_of_dicts = {}
    # query database for top 5 businesses for each category
    for cat_name in categories:
        # print "this is cat_name", cat_name
        cat_code = categories[cat_name]
        # select biz_id, avg_cat_review, num_revs from bizsentiments where cat_code='gltn' order by avg_cat_review desc, num_revs desc;
        biz_in_cat = BizSentiment.query.filter(BizSentiment.cat_code==cat_code)
        top_rated = biz_in_cat.order_by(BizSentiment.avg_cat_review.desc())
        top_five = top_rated.order_by(BizSentiment.num_revs.desc()).limit(5).offset(0).all()
        # print "this is top_five", top_five
        # create a list of dictionaries for the category
        top_five_list = []
        biz_rank = 1
        for biz in top_five:
            # use backref to get name and lat/long info
            name = biz.biz.name
            lat = biz.biz.lat
            lng = biz.biz.lng

            # make dictionary
            biz_dict = {'biz_id': biz.biz_id,
                        'name': name,
                        'avg_cat_review': biz.avg_cat_review,
                        'lat': lat,
                        'lng': lng,
                        'z_index': biz_rank}
            top_five_list.append(biz_dict)
            biz_rank += 1

        # update category dictionary and append to list of dicts
        # cat_dict = {cat: top_five_list}
        # print "this is top_five_list", top_five_list
        data_list_of_dicts[cat_code] = top_five_list
        print data_list_of_dicts

    return jsonify(data_list_of_dicts)


@app.route('/biz/<cat>/<int:n>/<int:oset>/next.json')
def get_next_n_results(cat, n, oset):
    """Returns JSON for a particular category containing next n results"""
    # remove 'unknown' category from list of categories
    cat_code = cat  # category
    # lim_results = n  # limit
    # oset_results = oset  # offset
    data_list_of_dicts = {}
    # select biz_id, avg_cat_review, num_revs from bizsentiments where cat_code='gltn' order by avg_cat_review desc, num_revs desc;
    biz_in_cat = BizSentiment.query.filter(BizSentiment.cat_code==cat_code)
    top_rated = biz_in_cat.order_by(BizSentiment.avg_cat_review.desc())
    next_n = top_rated.order_by(BizSentiment.num_revs.desc()).limit(n).offset(oset).all()
    # create a list of dictionaries for the category
    next_n_list = []
    biz_rank = 1
    for biz in next_n:
        # use backref to get name and lat/long info
        name = biz.biz.name
        lat = biz.biz.lat
        lng = biz.biz.lng

        # make dictionary
        biz_dict = {'biz_id': biz.biz_id,
                    'name': name,
                    'avg_cat_review': biz.avg_cat_review,
                    'lat': lat,
                    'lng': lng,
                    'z_index': biz_rank}
        next_n_list.append(biz_dict)
        biz_rank += 1

    data_list_of_dicts[cat_code] = next_n_list
    return jsonify(data_list_of_dicts)


@app.route('/search', methods=['GET'])
def search_bar_results():
    """Page displaying results of search bar search."""

    # get search form inputs
    # address
    search_loc = request.args.get("search-loc")
    if not search_loc:
        search_loc = 'Palo Alto, CA'

    parsed_loc = re.findall("([\w\s]+),\s(\w+)", search_loc)
    city = parsed_loc[0][0]
    state = parsed_loc[0][1]

    # TODO: FIND NEARBY CITY...
    # check if city in db...

    # search terms
    search_terms = request.args.get("search-terms")
    search_terms = search_terms.split()

    return render_template('search.html', google_maps_key=google_maps_key, cat_list=CAT_LISTS, state=state, state_name=STATE_CODES[state], city=city, search_terms=search_terms)


@app.route('/search/<search_terms>/<search_loc>/search.json')
def query_search(search_terms, search_loc):
    """
    return json of biz results by search
    """
    categories = dict(CAT_CODES)
    print "this is search_terms in search.json", search_terms
    print "this is search_loc in search.json", search_loc
    parsed_loc = re.findall("([\w\s]+),\s(\w+)", search_loc)
    city = parsed_loc[0][1]
    state = parsed_loc[0][1]

    search_terms = search_terms.split()
    clauses = and_( * [PlatePalReview.text.like('%'+ term + '%') for term in search_terms])

    categories = dict(CAT_CODES)

    # query database
    data_list_of_dicts = {}
    # query database for top 5 businesses for each category
    for cat_name in categories:
        cat_code = categories[cat_name]

        # select distinct Biz.biz_id, Biz.name, Biz.city from Biz
        # join Revcats on biz.biz_id = revcats.biz_id
        # join Reviews on revcats.review_id = reviews.review_id
        # where revcats.cat_code = 'gltn' and biz.city='Palo Alto';
        state_biz = db.session.query(PlatePalBiz).join(ReviewCategory).join(PlatePalReview).filter(PlatePalBiz.state==state)
        city_biz = state_biz.filter(PlatePalBiz.city==city).filter(clauses)
        # city_biz = state_biz.filter(PlatePalBiz.city==city).filter(PlatePalReview.text.like('%'+search_term+'%'))

        if cat_code != 'unkn':
            city_biz_cat = city_biz.filter(ReviewCategory.cat_code==cat_code).all()
        else:
            city_biz_cat = city_biz.all()

        cat_list = []
        for biz in city_biz_cat:
            biz_dict = {'biz_id': biz.biz_id,
                        'name': biz.name,
                        'address': biz.address,
                        'city': biz.city,
                        'state': biz.state,
                        'lat': biz.lat,
                        'lng': biz.lng,
                        'is_open': biz.is_open,
                        'photo_url': biz.photo_url
                        }
            # if biz_dict not in cat_list:
            cat_list.append(biz_dict)

        data_list_of_dicts[cat_code] = cat_list

    return jsonify(data_list_of_dicts)


@app.route('/biz')
def show_biz_general():
    return render_template('biz.html')


@app.route('/biz/<biz_id>')
def show_biz_details(biz_id):
    """Displays details for individual business."""

    biz = PlatePalBiz.query.get(biz_id)

    # TODO: Get average Yelp stars of business

    stars = [r.yelp_stars for r in biz.reviews if r.yelp_stars is not None]
    # print "This is stars", stars
    if stars:
        avg_star = float(sum(stars)) / len(stars)
        num_rev = len(stars)
    else:
        avg_stars = 0
        num_rev = 0

    # TODO: Get aggregate sentiment scores of business
    sen_scores = {}
    for cat in CAT_CODES_ID:
        cat_sen = [rc.agg_sen_score for rc in biz.sentiments if rc.cat_code == cat and rc.agg_sen_score is not None]
        # print "This is cat_sen", cat_sen
        if cat_sen:
            avg_cat_sen = float(sum(cat_sen)) / len(cat_sen)
        else:
            avg_cat_sen = 0
        sen_scores[cat] = avg_cat_sen

    # use subqueries to get reviews, user who wrote review, sen_score of review
    # get all reviews for the business
    reviews = PlatePalReview.query.filter(PlatePalReview.biz_id==biz_id).all()
    cat_reviews = {}
    for cat in CAT_CODES_ID:
        cat_reviews[cat] = []
    # get revcats
    for review in reviews:
        # query for revcats
        review.revcat
        if review.revcat != []:
            cat_reviews[review.revcat[0].cat_code].append(review)

    for cat in CAT_CODES_ID:
        if cat_reviews[cat] != []:
            for i, review in enumerate(cat_reviews[cat]):
                # Query for User name
                user = review.yelp_user
                # Query for revcat
                revcats = review.revcat
                # note that revcats is a list (review, user, [revcat, revcat...])
                cat_reviews[cat][i] = (review, user, revcats)

    # biz_query = db.session.query(PlatePalReview, YelpUser, ReviewCategory).join(YelpUser).join(ReviewCategory).filter(PlatePalReview.biz_id==biz_id, ReviewCategory.cat_code==cat).all()

    return render_template("biz.html", biz=biz, avg_star=avg_star, sen_scores=sen_scores, num_rev=num_rev, cat_reviews=cat_reviews, cats=CAT_CODES_ID)


@app.route('/biz/<biz_id>/add-review', methods=['POST'])
def update_business_review():
    # this route is only accessed by a logged-in user

    # get the biz_id (hidden submit)
    biz_id = int(request.form.get('biz-id'))

    # get user's rating
    user_review = request.form.get('user-review')

    user_id = session['user_id']

    # TODO


    return redirect("/biz/" + str(biz_id))


@app.route('/login-form')
def show_login_form():
    return render_template("login.html")


@app.route('/login-process')
def process_login():
    #TODO update, look into Flask login
    return "<h1>Nothing here</h1>"


@app.route('/profile/<int:user_id>')
def show_user_page(user_id):
    # TODO: update
    return render_template('profile.html', user=user_id)


@app.route('/logout')
def process_logout():
    # remove user id from the session
    del session['user_id']

    # create flash message "logged out"
    flash("Successfully logged out.")

    # redirect to homepage
    return redirect('/')


@app.route('/<state>/<city>/geocode.json')
def geocode_city_state(city, state):
    """
    return json of lat/long for a city, state

    used to center map for /state/city/city.html
    """
    # geolocator = Nominatim()
    # location = geolocator.geocode(city + ", " + state)
    # query db for similar city
    city_entry = City.query.filter(City.state==state, City.city.like('%'+city+'%')).first()

    return jsonify({'lat': city_entry.lat, 'lng': city_entry.lng})

@app.route('/analytics.json')
def get_force_data():
    #  sqlite> select DISTINCT biz.name, biz.city, revcats.cat_code
    # ...> FROM biz
    # ...> JOIN Reviews on Reviews.biz_id = biz.biz_id
    # ...> JOIN Revcats on Revcats.review_id = reviews.review_id
    # ...> WHERE Biz.city in ('Berkeley', 'Palo Alto', 'Menlo Park', 'Stanford')
    # ...> AND Biz.state = 'CA';

    # Define NODES
    #   // TODO: use JSON to get node
    #  var nodes = [
    #      {name: 'Meggie', advisor: 'Meggie'}, // 0
    #      {name: 'Cynthia', advisor: 'Cynthia'}, // 1

    cities_list = ['Berkeley', 'Menlo Park', 'Palo Alto', 'Stanford']

    nodes = []
    nodes_index = []
    n_index = 0
    # Add categories to nodes
    for cat in CAT_NAMES_ID[:len(CAT_NAMES_ID)-1]:
        nodes.append({'name': cat, 'parent': cat})
        nodes_index.append({cat: n_index})
        n_index += 1
    # add cities to nodes
    for city in cities_list:
        nodes.append({'name': city, 'parent': city})
        nodes_index.append({city: n_index})
        n_index += 1

    # Select businesses in cities list that have a category code (Inner Join)
    QUERY = """
    SELECT DISTINCT Biz.name, Biz.city, Biz.biz_id FROM Biz
    JOIN Reviews on Reviews.biz_id = Biz.biz_id
    JOIN Revcats on Revcats.review_id = Reviews.review_id
    WHERE Biz.city in ('Berkeley', 'Palo Alto', 'Menlo Park', 'Stanford')
    AND Biz.state = 'CA';
    """
    cursor = db.session.execute(QUERY)
    biz = cursor.fetchall()

    for b in biz: # b[0] = name, b[1] = city, b[2] = biz_id
        b_idict = {}
        b_odict = {b[0]: b_idict}
        nodes.append({'name': b[0], 'parent': b[1]})
        
        # append biz-city pair to index dictionary
        b_idict[b[1]] = n_index
        n_index += 1
        
        # get cat codes for that business
        QUERY = """
        SELECT Revcats.cat_code FROM Revcats
        JOIN Reviews on Reviews.review_id = Revcats.review_id
        JOIN Biz on Biz.biz_id = Reviews.biz_id
        WHERE Biz.biz_id = :biz_id
        """
        cursor = db.session.execute(QUERY, {'biz_id': b[2]})
        cats = cursor.fetchall()
        # import pdb; pdb.set_trace()
        for cat in cats:
            cat_name = CAT_DICT[cat[0]]
            nodes.append({'name': b[0], 'parent': cat_name})
            b_idict[cat] = n_index
            n_index += 1
    nodes_index.append(b_odict)
    # define LINKS
    # link business to city
    # link business to category
    return jsonify(nodes)

def find_nearby_cities(city, state, x_miles):
    """Given a city (city, state), return a list of cities within x miles."""
    # query db for city id
    city_obj = City.query.filter(City.city==city, City.state==state).first()
    if city_obj:
        city_id = city_obj.city_id

        # query db for list of nearby cities within x miles
        nearby_cities = db.session.query(CityDistance.city2_id).filter(CityDistance.city1_id==city_id).filter(CityDistance.miles < x_miles).all()

        nearby_cities_list = []
        # lookup city names for nearby cities
        for nearby_city in nearby_cities:
            nearby_name = db.session.query(City.city).filter(City.city_id==nearby_city[0]).first()
            nearby_cities_list.append(nearby_name)
    else:
        nearby_cities_list = []

    return nearby_cities_list

# TODO -- for adding a new review
def get_sentiment_score(doc):
    url = "http://text-processing.com/api/sentiment/"

    payload = {'text': doc}

    # make API call
    r = requests.post(url, data=payload)

    # load JSON from API call 
    result = json.loads(r.text)

    # pull sentiment score
    sen_score = result['probability']['pos']

    # store in database...
    return sen_score

if __name__ == "__main__":
    # We have to set debug=True here, since it has to be True at the point
    # that we invoke the DebugToolbarExtension
    app.debug = True

    connect_to_db(app)

    # Use the DebugToolbar
    DebugToolbarExtension(app)

    app.run()

    get_force_data()