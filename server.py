from jinja2 import StrictUndefined

from flask import Flask, render_template, redirect, request, flash, session, jsonify
from flask_debugtoolbar import DebugToolbarExtension

from model import YelpBiz, YelpUser, YelpReview
from model import PlatePalBiz, PlatePalUser, PlatePalReview
from model import UserList, ListEntry
from model import Category, ReviewCategory, BizSentiment
from model import City, CityDistance, CityDistCat
from model import connect_to_db, db

from collections import defaultdict
from collections import OrderedDict
from sqlalchemy import distinct
from model import CAT_CODES
from statecodes import STATE_CODES
from datetime import datetime

import re
import os
import requests
import json

# from geopy.geocoders import Nominatim

CAT_CODES_ID = ['gltn', 'vgan', 'kshr', 'algy', 'pleo', 'unkn']
CAT_NAMES_ID = ['Gluten-Free', 'Vegan', 'Kosher', 'Allergies', 'Paleo', 'Feeling Lucky']
CAT_DICT = {CAT_CODES_ID[n]: CAT_NAMES_ID[n] for n in range(len(CAT_CODES_ID))}
CAT_IDICT = {CAT_NAMES_ID[n]: CAT_CODES_ID[n] for n in range(len(CAT_CODES_ID))}
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


@app.route('/state.html')
def display_states():
    """Displays all cities with reviews in PlatePal categories by state."""
    # query db for cities by state
    QUERY="""
        SELECT DISTINCT Biz.city, Biz.state from Biz
        INNER JOIN Reviews on reviews.biz_id = Biz.biz_id
        INNER JOIN revcats on revcats.review_id = reviews.review_id
        WHERE revcats.cat_code in ('gltn', 'vgan', 'pleo', 'kshr', 'algy')
        ORDER BY Biz.state;"""

    results = db.session.execute(QUERY).fetchall()

    # create a dictionary where state abbrev = keys and values
    # are a list of cities in the state
    locations = defaultdict(list)
    for result in results:
        locations[result[1]].append(result[0])
    print locations
    return render_template('state.html', locations=locations)

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
    return render_template('state-old.html', google_maps_key=google_maps_key, cat_list=CAT_LISTS, state=state, state_name=STATE_CODES[state], state_cities=state_cities)


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
    print "tHIS IS nearby_cities", nearby_cities

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



@app.route('/<state>/<city>/sunburst')
def show_zoomable_sunburst_labels(state="CA", city="Berkeley"):
    """Analytics page housing sentiment analysis info and D3"""
    # QUERY = """
    # SELECT DISTINCT state from Cities;
    # """
    # states = db.session.execute(QUERY).fetchall()
    # states = sorted(states)
    states = [['CA']]

    # QUERY = """
    # SELECT DISTINCT city from Cities;
    # """
    # cities = db.session.execute(QUERY).fetchall()
    cities = [['Berkeley'],['Claremont'], ['La Jolla'],['Los Angeles'],
              ['Palo Alto'], ['Pasadena'], ['San Diego'], ['San Luis Obispo']]
    return render_template("zoomable-sunburst-labels.html",
                           state=state, city=city,
                           states=states, cities=cities)

@app.route('/sunburst-form', methods=['GET'])
def update_zoomable_sunburst_labels():
    """Analytics page housing sentiment analysis info and D3"""
    city = request.args.get("cityname")
    state = request.args.get("statename")
    return redirect("/" + state + "/" + city + "/sunburst")



@app.route('/sunburst-labels')
def show_sunburst_labels():
    """Analytics page housing sentiment analysis info and D3"""
    return render_template("sunburst-labels.html")


@app.route('/sunburst-basic')
def show_sunburst_basic():
    """Analytics page housing sentiment analysis info and D3"""
    return render_template("sunburst.html")


# TODO: implement better treemap
# from collections import defaultdict
# def tree():
#     """Helper function for generating tree-like JSON"""
#     return defaultdict(tree)


@app.route('/<selected_state>/<selected_city>/sunburst.json', methods=['GET', 'POST'])
def get_sunburst_data(selected_state, selected_city):
    """Make a tree structure of JSON, like mbostock's flare.json

    root = City -> Category -> Business -> Review -> Sentiment Score
    """
    print "IN SUNBURST.JSON"
    # selected_city = request.form.get('cityname')
    state = selected_state
    cities_list = [selected_city]
    print cities_list
    # cities_list = ['Berkeley']

    arc_size = 10000

    for city in cities_list:
        # get number of reviews in city by category (for sizing children)
        QUERY = """
        SELECT count(*), Revcats.cat_code from revcats
        JOIN Biz on Biz.biz_id = Revcats.biz_id
        WHERE Biz.city = :city and Biz.state = :state
        GROUP BY Revcats.cat_code;
        """
        cursor = db.session.execute(QUERY, {'city': city, 'state': state})
        revcounts = cursor.fetchall()

        revcount_dict = {}
        for revcount in revcounts:
            revcount_dict[str(revcount[1])] = revcount[0]
        total_reviews_in_city = sum(revcount_dict.values())

        # make tree(s)
        city_tree = dict()
        city_tree['name'] = city
        city_tree['children'] = []
        for cat in revcount_dict.keys():
            print "this is cat", cat
            cat_tree = dict()
            cat_tree['name'] = CAT_DICT[cat]
            cat_tree['children'] = []
            # query for businesses in category
            QUERY = """
            SELECT DISTINCT Biz.name, Biz.biz_id FROM Biz
            INNER JOIN Reviews on Reviews.biz_id = Biz.biz_id
            INNER JOIN Revcats on Revcats.review_id = Reviews.review_id
            WHERE Revcats.cat_code = :cat_code
            AND Biz.city = :city AND Biz.state = 'CA';"""
            cursor = db.session.execute(QUERY, {'cat_code': cat, 'city': city})
            businesses = cursor.fetchall()

            # proportional of all reviews in city in current category
            cat_weight = (1.0 * revcount_dict[cat] ) / total_reviews_in_city

            for business in businesses:
                # handle accented characters
                if u'\xe9' in business[0]:
                    biz_name = business[0].replace(u'\xe9', u'e')
                else:
                    biz_name = business[0]
                biz_id = business[1]
                biz_tree = dict()
                biz_tree['name'] = biz_name
                biz_tree['size'] = 100

                # # TODO: TOO MANY REVIEWS TO SHOW THIS LAYER
                # biz_tree['children'] = []
                # # query for reviews in catergory in business
                # QUERY = """
                # SELECT yelpUsers.name, Revcats.review_id, Revcats.sen_score, Revcats.revcat_id FROM Revcats
                # INNER JOIN Reviews on Reviews.review_id = Revcats.review_id
                # INNER JOIN yelpUsers on Reviews.yelp_user_id = yelpUsers.user_id
                # WHERE Revcats.cat_code = :cat_code
                # AND Revcats.biz_id = :biz_id"""
                # cursor = db.session.execute(QUERY, {'biz_id': biz_id, 'cat_code': cat})
                # reviews = cursor.fetchall()
                # # get total sentiment score for calculating size
                # biz_sen_total = 0
                # for review in reviews:
                #     # if review does not have sentiment, score it and add to db.
                #     if not review[2]:
                #         # query for review text
                #         doc = PlatePalReview.query.filter(PlatePalReview.review_id==review[1]).one()
                #         if doc:
                #             sen_score = get_sentiment_score(doc)
                #             revcat = ReviewCategory.query.filter(ReviewCategory.revcat_id==review[3]).one()
                #             revcat.sen_score = sen_score
                #             db.session.commit()
                #     else:
                #         revcat_sen = review[2]
                #     # update total
                #     biz_sen_total += revcat_sen
                # # create review tree
                # for review in reviews:
                #     user_name = review[0]
                #     if not review[2]:
                #         # query for review sentiment in database (updated above)
                #         revcat = ReviewCategory.query.filter(ReviewCategory.revcat_id==review[3]).one()
                #         sen_score = revcat.sen_score
                #     else:
                #         sen_score = review[2]

                #     rev_size = ( sen_score / biz_sen_total ) * cat_weight * arc_size
                #     review_tree = {"name": user_name, "size": rev_size}
                #     # add review to business children
                #     biz_tree['children'].append(review_tree)
                # add biz to category children
                cat_tree['children'].append(biz_tree)
            # add category to city children
            city_tree['children'].append(cat_tree)

            # add trees for categories not in city --> this was showing results twice ...
            # set1 = set(CAT_DICT.keys())
            # set2 = set(revcount_dict.keys())
            # print set1
            # print set2
            # print set1.difference(set2)
            # for item in set1.difference(set2):
            #     print "this is cat in set difference", item
            #     cat_tree = dict()
            #     cat_tree['name'] = CAT_DICT[item]
            #     cat_tree['size'] = arc_size / 100
            #     city_tree['children'].append(cat_tree)
    print "this is city_tree['name']", city_tree['name']
    return jsonify(city_tree)


@app.route('/analytics')
def lab():
    """Analytics page housing sentiment analysis info and D3"""
    # return render_template("force-labeled.html")
        # QUERY = """
    # SELECT DISTINCT state from Cities;
    # """
    # states = db.session.execute(QUERY).fetchall()
    # states = sorted(states)
    states = [['CA']]

    # QUERY = """
    # SELECT DISTINCT city from Cities;
    # """
    # cities = db.session.execute(QUERY).fetchall()
    cities = [['Berkeley'],['Claremont'], ['La Jolla'],['Los Angeles'],
              ['Palo Alto'], ['Pasadena'], ['San Diego'], ['San Luis Obispo']]
    return render_template("analytics.html", state='CA', city='Berkeley', states=states, cities=cities)


@app.route('/<region>/force.json')
def get_force_data(region):
    """Generate JSON for Force Graph by region"""
    # region = 'socal'
    if region == 'norcal':
        cities_list = ['Berkeley', 'Palo Alto', 'Stanford']
    else:
        cities_list = ['Claremont', 'Los Angeles', 'Pasadena']

    cities_str = "("
    for place in cities_list:
        cities_str += place + ', '
    cities_str = cities_str.rstrip(',')
    cities_str += ')'

    links = []
    nodes = []
    nodes_index = {}
    n_index = 0
    # Add categories to nodes
    for cat in CAT_NAMES_ID[:len(CAT_NAMES_ID)-1]:
        # nodes.append({'name': cat, 'parent': cat})
        nodes.append({'name': cat, 'group': 1, 'value': 10})
        nodes_index[cat] = n_index
        n_index += 1
    # add cities to nodes
    for city in cities_list:
        # nodes.append({'name': city, 'parent': city})
        nodes.append({'name': city, 'group': 2, 'value': 15})
        nodes_index[city] = n_index
        n_index += 1
    print "this is nodes", nodes
    print "this is nodes_index", nodes_index
    # Select businesses in cities list that have a category code (Inner Join)
    if region == 'norcal':
        QUERY = """
        SELECT DISTINCT Biz.name, Biz.city, Biz.biz_id FROM Biz
        INNER JOIN Reviews on Reviews.biz_id = Biz.biz_id
        INNER JOIN Revcats on Revcats.review_id = Reviews.review_id
        WHERE Biz.city in ('Berkeley', 'Palo Alto', 'Stanford')
        AND Biz.state = 'CA';
        """
    else: 
        QUERY = """
        SELECT DISTINCT Biz.name, Biz.city, Biz.biz_id FROM Biz
        INNER JOIN Reviews on Reviews.biz_id = Biz.biz_id
        INNER JOIN Revcats on Revcats.review_id = Reviews.review_id
        WHERE Biz.city in ('Claremont', 'Los Angeles', 'Pasadena')
        AND Biz.state = 'CA';
        """
    cursor = db.session.execute(QUERY)
    biz = cursor.fetchall()
    for place in biz: # place[0] = name, place[1] = city, place[2] = biz_id
        b_dict = {}
        if u'\xe9' in place[0]:
            biz_name = place[0].replace(u'\xe9', u'e')
        else:
            biz_name = place[0]
        # nodes.append({'name': biz_name, 'parent': place[1]})
        nodes.append({'name': biz_name, 'group': 3, 'value': 5})

        # append biz-city pair to index dictionary
        b_dict[biz_name] = [n_index, place[1]]
        links.append({'source': nodes_index[place[1]], 'target': n_index, 'value': 1})
        if 'Stuffed Inn' in biz_name:
            print '-'*20
            print 'STUFFED INN'
            print 'Nodes'
            for node in nodes:
                print node
            print 'Links'
            for link in links:
                print link
            print '-'*20
        # get cat codes for that business
        QUERY = """
        SELECT DISTINCT Revcats.cat_code FROM Revcats
        INNER JOIN Reviews on Reviews.review_id = Revcats.review_id
        INNER JOIN Biz on Biz.biz_id = Reviews.biz_id
        WHERE Biz.biz_id = :biz_id
        """
        cursor = db.session.execute(QUERY, {'biz_id': place[2]})
        cats = cursor.fetchall()
        # import pdb; pdb.set_trace()
        for c in cats:
            cat_name = CAT_DICT[c[0]]
            # b_dict[cat_name] = n_index
            b_dict[biz_name].append(cat_name)
            # n_index += 1
            links.append({'source': nodes_index[cat_name], 'target': n_index, 'value': 1})
        # keep track of nodex
        nodes_index[biz_name] = b_dict
        # make links
        n_index += 1
    # build outer dictionary
    json_dict = {'nodes': nodes, 'links': links}
    
    return jsonify(json_dict)


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
            nearby_name = db.session.query(City.city, City.state).filter(City.city_id==nearby_city[0]).first()

            print "this is nearby_name[0]", nearby_name[0]
            # check if city has any reviews in revcats
            QUERY="""
            SELECT * from Revcats
            JOIN Reviews on Reviews.review_id = Revcats.review_id
            JOIN Biz on Biz.biz_id = Reviews.biz_id
            WHERE Biz.city = :city
            """
            revcats = db.session.execute(QUERY, {'city': nearby_name[0]}).fetchall()
            if revcats:
                nearby_cities_list.append(nearby_name)
    else:
        nearby_cities_list = []

    return nearby_cities_list


# @app.route('/scatterplot.tsv')
def get_scatter_data():
    QUERY="""
    SELECT revcat_id, revcats.review_id, reviews.review_date, revcats.sen_score, reviews.yelp_stars
    FROM revcats
    JOIN reviews on reviews.review_id = revcats.review_id
    WHERE reviews.biz_id = 2171
    ORDER BY reviews.review_date"""

    sen_scores_by_date = db.session.execute(QUERY).fetchall()

    # calculate months from date
    date_format = "%Y-%m-%d"

    zero_date = sen_scores_by_date[0][2]
    zero_date = datetime.strptime(zero_date, date_format)
    # import pdb; pdb.set_trace()
    with open("./static/tsv/scatterplot.tsv", "w") as record_file:
        record_file.write("reviewDate  timeDelta  sentimentScore  stars\n")
        for entry in sen_scores_by_date:
            entry_date = datetime.strptime(entry[2], date_format)    
            delta = entry_date - zero_date
            entry_months = delta.days
            record_file.write(entry[2] +"  "+str(entry[3])+"  "+ str(entry[4])+"\n")
    record_file.close()
    return


@app.route('/scatterplot')
def show_scatter_plot():
    return render_template("scatterplot.html")


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

    get_scatter_data()
    # get_force_data()