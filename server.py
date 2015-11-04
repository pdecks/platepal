from jinja2 import StrictUndefined

from flask import Flask, render_template, redirect, request, flash, session
from flask_debugtoolbar import DebugToolbarExtension

from model import YelpBiz, YelpUser, YelpReview
from model import PlatePalBiz, PlatePalUser, PlatePalReview
from model import UserList, ListEntry
from model import Category, ReviewCategory, BizSentiment
from model import connect_to_db, db


app = Flask(__name__)

# Required to use Flask sessions and the debug toolbar
app.secret_key = "ABC"

# Normally, if you use an undefined variable in Jinja2, it fails silently.
# This is horrible. Fix this so that, instead, it raises an error.
app.jinja_env.undefined = StrictUndefined

@app.route('/')
def index():
    """Homepage."""

    return render_template('home.html')
    # "<html><body>Placeholder for the homepage.</body></html>"


@app.route('/biz/<int:biz_id>')
def show_biz_details(biz_id):
    """Displays details for individual business."""

    biz = PlatePalBiz.query.get(biz_id)

    user_id = session.get("user_id")

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
    #TODO update
    username = request.args.get('username')
    password = request.args.get('password')
    # query for username in database

    #TODO update
    # if user = None, add user to database
    user = User.query.filter(User.email == username).first()
    # print "This is user after line 74: %s" % user
    if user == None:
        age = ''
        zipcode = ''
        user = User(email=username, password=password, age=age, zipcode=zipcode)
        # print "This is user after line 79: %s" % user
        db.session.add(user)
        db.session.commit()

        # User.query.filter(User.email == username)
        user = User.query.filter(User.email == username).all()
        # user = user[0]
        # print "This is the user after line 85: %s" % user
        # print "This is password: %s" % password
        # print "This is user.password: %s" % user.password

        # # TODO: ways to get fancy: add modal window, registration page, etc.

    # user exists, check pw
    # log in user if password matches user pw
    if user.password == password:
        # add user id to session
        session['user_id'] = user.user_id
        # create flash message 'logged in'
        flash("Login successful.")

    else:
        # display alert for incorrect login information
        flash("Incorrect login information. Please try again.")
        # good place to use AJAX in the future!
        return redirect('/login-form')

    # redirect to homepage
    # return redirect('/users/<user_id>')
    return redirect('/profile/' + str(session[user_id]))


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