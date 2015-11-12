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
import os

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


@app.route('/<state>/<city>/city.html')
def city_in_state_page(state, city):
    """City Homepage"""

    # get nearby cities list
    nearby_miles = 50 # find cities within 50 miles of current city
    nearby_cities = find_nearby_cities(city, state, nearby_miles)

    return render_template('city.html', google_maps_key=google_maps_key, cat_list=CAT_LISTS, state=state, state_name=STATE_CODES[state], city=city, nearby_cities=nearby_cities)


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
        # join Revcats on biz.biz_id = revcats.biz_id
        # where revcats.cat_code = 'gltn' and biz.city='Palo Alto';
        state_biz = db.session.query(PlatePalBiz).join(ReviewCategory).filter(PlatePalBiz.state==state)
        city_biz = state_biz.filter(PlatePalBiz.city==city)

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


@app.route('/search')
def search_bar_results():
    """Page displaying results of search bar search."""
    return render_template('search.html')


@app.route('/biz')
def show_biz_general():
    return render_template('biz.html')

@app.route('/biz/<biz_id>')
def show_biz_details(biz_id):
    """Displays details for individual business."""

    biz = PlatePalBiz.query.get(biz_id)

    # TODO: Get average Yelp stars of business

    stars = [r.stars for r in biz.reviews]
    avg_star = float(sum(stars)) / len(stars)

    # TODO: Get aggregate sentiment scores of business
    sentiment = [cat1, cat2, cat3, cat4, cat5, cat6]
    sen_scores = []
    for sen in sentiment:
        cat_sen = [rc.sen_score for rc in biz.sentiments if rc.cat_code == sentiment]
        avg_cat_sen = float(sum(cat_sen)) / len(cat_sen)

    # # query for movie's ratings, return list of tuples [(user.email, ratings.score), ...]
    # # add user_id
    # QUERY = """
    #         SELECT Users.email, Users.user_id, Ratings.score
    #         FROM Ratings
    #         JOIN Movies ON Movies.movie_id = Ratings.movie_id
    #         JOIN Users ON Users.user_id = Ratings.user_id
    #         WHERE Movies.movie_id = :movie_id
    #         ORDER BY Users.user_id;
    #         """
    # cursor = db.session.execute(QUERY, {'movie_id': movie_id})
    # ratings = cursor.fetchall()

    # movies = Movie.query.order_by(Movie.movie_title).all()
    # return render_template("movie-details.html", movie=movie, ratings=ratings)

    return render_template("biz.html", biz=biz)


@app.route('/biz/<biz_id>/add-review', methods=['POST'])
def update_business_review():
    # this route is only accessed by a logged-in user

    # get the biz_id (hidden submit)
    biz_id = int(request.form.get('biz-id'))

    # get user's rating
    user_review = request.form.get('user-review')

    user_id = session['user_id']

    # TODO
    # query Rating to see if user has already rated
    # if rating = None, add rating to database
    # rating = Rating.query.filter(Rating.user_id == user_id, Rating.movie_id == movie_id).first() ### SOMETHING IS GOING WRONG HERE ...
    # print "rating: %s" % rating

    # if rating == None:
    #     # add value in ratings database
    #     rating = Rating(movie_id=movie_id, user_id=user_id, score=user_rating)
    #     print "This is rating: movie_id %s user_id %s score %s" % (rating.movie_id, rating.user_id, rating.score)
    #     db.session.add(rating)
    #     db.session.commit()
    #     flash("Your rating has been sucessfully added!")
    # else: # update rating
    #     QUERY = """
    #     UPDATE Ratings
    #     SET score=:score
    #     WHERE user_id=:user_id AND movie_id=:movie_id;
    #     """
    #     cursor = db.session.execute(QUERY, {'user_id': user_id, 'movie_id': movie_id, 'score': user_rating})
    #     db.session.commit()
    #     flash("Your rating has been sucessfully updated!")

    return redirect("/biz/" + str(biz_id))


@app.route('/login-form')
def show_login_form():
    return render_template("login.html")

@app.route('/login-process')
def process_login():
    #TODO update, look into Flask login
    # username = request.args.get('username')
    # password = request.args.get('password')
    # # query for username in database

    # #TODO update
    # # if user = None, add user to database
    # user = User.query.filter(User.email == username).first()
    # # print "This is user after line 74: %s" % user
    # if user == None:
    #     age = ''
    #     zipcode = ''
    #     user = User(email=username, password=password, age=age, zipcode=zipcode)
    #     # print "This is user after line 79: %s" % user
    #     db.session.add(user)
    #     db.session.commit()

    #     # User.query.filter(User.email == username)
    #     user = User.query.filter(User.email == username).all()
    #     # user = user[0]
    #     # print "This is the user after line 85: %s" % user
    #     # print "This is password: %s" % password
    #     # print "This is user.password: %s" % user.password

    #     # # TODO: ways to get fancy: add modal window, registration page, etc.

    # # user exists, check pw
    # # log in user if password matches user pw
    # if user.password == password:
    #     # add user id to session
    #     session['user_id'] = user.user_id
    #     # create flash message 'logged in'
    #     flash("Login successful.")

    # else:
    #     # display alert for incorrect login information
    #     flash("Incorrect login information. Please try again.")
    #     # good place to use AJAX in the future!
    #     return redirect('/login-form')

    # # redirect to homepage
    # # return redirect('/users/<user_id>')
    # return redirect('/profile/' + str(session[user_id]))
    pass

@app.route('/profile/<int:user_id>')
def show_user_page(user_id):
    # TODO: update
    # # user_id = session['user_id']
    # # user_id = user_id #pdecks@me.com
    # QUERY = """
    #         SELECT Movies.movie_title, Ratings.score
    #         FROM Ratings
    #         JOIN Movies ON Movies.movie_id = Ratings.movie_id
    #         JOIN Users ON Users.user_id = Ratings.user_id
    #         WHERE Users.user_id = :user_id;
    #         """
    # cursor = db.session.execute(QUERY, {'user_id': user_id})
    # movies = cursor.fetchall()
    # user = User.query.filter(User.user_id == user_id).one()

    # TODO: Allow user to edit information on profile page if logged in

    return render_template('profile.html', user=user)


@app.route('/logout')
def process_logout():
    # remove user id from the session
    del session['user_id']

    # create flash message "logged out"
    flash("Successfully logged out.")

    # redirect to homepage
    return redirect('/')


# @app.route('/users/<int:user-id>')
# def show_user():
#     return

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



if __name__ == "__main__":
    # We have to set debug=True here, since it has to be True at the point
    # that we invoke the DebugToolbarExtension
    app.debug = True

    connect_to_db(app)

    # Use the DebugToolbar
    DebugToolbarExtension(app)

    app.run()