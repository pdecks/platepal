"""Utility file to seed PlatePalReview classifications."""

import pdlinclass


from model import PlatePalBiz, PlatePalUser, PlatePalReview

from model import connect_to_db, db
from server import app

pickle_path = pdlinclass.pickle_path_SVC

# revive the linearSVC classifier
pipeline_clf = pdlinclass.revives_pipeline(pickle_path)


def classify_pp_review(review_text):
    """Returns a predicted category from the text of a review."""

    new_doc = [review_text]
    new_doc_category_id_pipeline = pipeline_clf.predict(new_doc)
    new_doc_category_pipeline = pdlinclass.get_category_name(new_doc_category_id_pipeline)

    if new_doc_category_pipeline == 'gluten':
        cat_code = 'gltn'
    elif new_doc_category_pipeline == 'unknown':
        cat_code = 'unkn'
    else:
        print 'Something went wrong with the classification.'

    return cat_code


def classify_existing_entries():
    """Updates PlatePalReviews database with cat_codes for all seeded reviews."""
    reviews = PlatePalReview.query.filter(PlatePalReview.cat_code.is_(None)).all()
    # reviews = PlatePalReview.query.limit(10).all()
    # cat_code = classify_pp_review(reviews.text)

    for review in reviews:
        cat_code = classify_pp_review(review.text)
        print cat_code

        # update cat_code entry in row
        review.cat_code = cat_code
        db.session.commit()

    return


if __name__ == "__main__":
    connect_to_db(app)

    # In case tables haven't been created, create them
    db.create_all()

    # Import different types of data
    classify_existing_entries()

    # Seed PlatePalBiz and PlatePalReview
    # load_pp_biz(bdf)
    # load_pp_reviews(rdf)
