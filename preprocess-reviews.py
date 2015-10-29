"""
Preprocesses .txt review files for importing into sklearn
as a dataset of bunches. Splits the text on pipes and rewrites the files,
leaving only the review text.

by Patricia Decker 10/29/2015
"""

# from os import environ
from os.path import dirname
from os.path import join
from os.path import isdir
from os import listdir
from os import utime
import pdb

# define a function to create new files for storing dictionary strings
def touch(fname, times=None):
    with open(fname, 'a'):
        utime(fname, times)


# directory containing review category folders
container_path = '/Users/pdecks/hackbright/project/Yelp/mvp/pdecks-reviews/'

# subfolders in directory also serve as category (target) names in bunch
# ex. categories = ['bad', 'excellent', 'good', 'limited', 'neutral', 'shady']
folders = [f for f in sorted(listdir(container_path))
           if isdir(join(container_path, f))]

# for specifying a subset of categories using variable 'categories' (not defined)
# if categories is not None:
#     folders = [f for f in folders if f in categories]

# iterate over the subdirectories
for label, folder in enumerate(folders):

    folder_path = join(container_path, folder)

    # create a list of document paths in the subdirectory
    documents = [join(folder_path, d)
                 for d in sorted(listdir(folder_path))]
    for d in documents:
        # ignore finder's .DS_Store
        if '.DS_Store' in d:
            continue

        # open the text file and save the review information to variables
        f = open(d, 'rw+')
        # pdb.set_trace()  # debugging
        data = f.read()
        review_data = data.split("|")
        rest_name = review_data[0]
        rest_stars = review_data[1]
        review_date = review_data[2]
        review_text = review_data[3]

        # reformat date to match Yelp JSON
        review_date_month = review_date[:2]
        review_date_day = review_date[2:4]
        review_date_year = review_date[4:]
        yelp_date = review_date_year + '-' + review_date_month + '-' + review_date_day

        f.close()

        # reopen the file and overwrite from the top of the file
        f = open(d, 'w')
        f.truncate()
        f.close()

        # reopen the file and write in only the review text
        f = open(d, 'r+')
        f.write(review_text)
        f.close()


        # # export dictionary as string to new file
        # line1 = "{'type': 'review', "
        # line2 = "'business_id': " + "'" + rest_name + "',"
        # line3 = "'user_id': " + 'greyhoundmama' + ","
        # line4 = "'stars': " + 'int(' + rest_stars + ")," 
        # line5 = "'text': " + "'" + review_text + "'," 
        # line6 = "'date': " + "'" + yelp_date + "'," 
        # line7 = "'votes': {'useful': 0, 'funny': 0, 'cool': 0}," 
        # line8 = "'target': " + "'" + folder + "'}"
        # review_dict_string = line1 + line2 + line3 + line4 + line5 + line6 + line7 + line8
        


        # # define new filename
        # new_f = d[:-4] + '-dict.txt'
        # touch(new_f)
        # new_file = open(new_f)
        # new_file.write(review_dict_string)
        # new_file.close()
