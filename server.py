from jinja2 import StrictUndefined

from flask import Flask, render_template, redirect, request, flash, session, jsonify
from flask_debugtoolbar import DebugToolbarExtension

from model import YelpBiz, YelpUser, YelpReview
from model import PlatePalBiz, PlatePalUser, PlatePalReview
from model import UserList, ListEntry
from model import Category, ReviewCategory, BizSentiment
from model import connect_to_db, db

from model import CAT_CODES

import os

CAT_CODES_ID = ['gltn', 'vgan', 'kshr', 'algy', 'pleo']
CAT_NAMES_ID = ['Gluten-Free', 'Vegan', 'Kosher', 'Allergies', 'Paleo']
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

    return render_template('home.html', google_maps_key=google_maps_key, cat_list=CAT_LISTS, limit_results=5, offset_results=5)


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

@app.route('/biz/<int:biz_id>')
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


@app.route('/biz/<int:biz_id>/add-review', methods=['POST'])
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

if __name__ == "__main__":
    # We have to set debug=True here, since it has to be True at the point
    # that we invoke the DebugToolbarExtension
    app.debug = True

    connect_to_db(app)

    # Use the DebugToolbar
    DebugToolbarExtension(app)

    app.run()