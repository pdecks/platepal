"""
Base IO code for all datasets
"""
# modified scikit-learn/sklearn/datasets/base.py
# modified by Paricia Decker for HB Independent Project
# 11/2015

# Copyright (c) 2007 David Cournapeau <cournape@gmail.com>
#               2010 Fabian Pedregosa <fabian.pedregosa@inria.fr>
#               2010 Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD 3 clause

import os
import csv
import sys
import shutil
from os import environ
from os.path import dirname
from os.path import join
from os.path import exists
from os.path import expanduser
from os.path import isdir
from os.path import splitext
from os import listdir
from os import makedirs

import numpy as np

# from ..utils import check_random_state


class Bunch(dict):
    """Container object for datasets
    Dictionary-like object that exposes its keys as attributes.
    >>> b = Bunch(a=1, b=2)
    >>> b['b']
    2
    >>> b.b
    2
    >>> b.a = 3
    >>> b['a']
    3
    >>> b.c = 6
    >>> b['c']
    6
    """

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __getstate__(self):
        return self.__dict__


def load_files(container_path, description=None, categories=None,
               encoding=None, decode_error='strict'):
    """Load text files **categories extracted from beginning of file
    (pipe-delimited)** (differs from original sklearn version)

    The individual file names are not important.

    This function does not try to extract features into a numpy array or
    scipy sparse matrix. In addition, if load_content is false it
    does not try to load the files in memory.

    To use text files in a scikit-learn classification or clustering
    algorithm, you will need to use the `sklearn.feature_extraction.text`
    module to build a feature extraction transformer that suits your
    problem.

    Specify the encoding of the text using the 'encoding' parameter.
    For many modern text files, 'utf-8' will be the correct encoding. If
    you leave encoding equal to None, then the content will be made of bytes
    instead of Unicode, and you will not be able to use most functions in
    `sklearn.feature_extraction.text`.

    Parameters
    ----------
    container_path : string or unicode
        Path to the main folder
    description: string or unicode, optional (default=None)
        A paragraph describing the characteristic of the dataset: its source,
        reference, etc.
    categories : A collection of strings or None, optional (default=None)
        If None (default), load all the categories.
        If not None, list of category names to load (other categories ignored).
    encoding : string or None (default is None)
        If None, do not try to decode the content of the files (e.g. for
        images or other non-text content).
        If not None, encoding to use to decode text files to Unicode if
        load_content is True.
    decode_error: {'strict', 'ignore', 'replace'}, optional
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. Passed as keyword
        argument 'errors' to bytes.decode.
    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are: either
        data, the raw text data to learn, or 'filenames', the files
        holding it, 'target', the classification labels (integer index),
        'target_names', the meaning of the labels, and 'DESCR', the full
        description of the dataset.
    """
    target = []
    target_names = []
    filenames = []
    if not categories:
        categories = ['gltn', 'vgan', 'kshr', 'algy', 'pleo', 'unkn']
    # target index = [0, 1, 2, 3, 4, 5]
    # all files will be in one folder: container_path = ./data/random_forest

    target_names.extend(categories)

    # create list of all documents in container_path
    documents = [join(folder_path, d)
                 for d in sorted(listdir(container_path))]

    # TODO added by pd
    data = []
    for i, filename in enumerate(documents):
        with open(filename, 'rb') as f:
            review_data = f.read()
            # split on pipes ...
            review = review_data.split('|')
            # 17776|975|The Wine Cellar|2006-08-22|There is ...
            review_id = review[0]
            biz_id = review[1]
            biz_name = review[2]
            review_date = review[3]
            review_text = review[4]

            data.append(review_text)

            # get document labels
            # 'gltn'|'vgan'|'kshr'|'algy'|'pleo'|'unkn'|# stars|review_id|biz_id|biz_name|review_date|text
            # 1|1|0|0|0|17776|975|The Wine Cellar|2006-08-22|There is a great ( think dollar store) place ...
            # if category binary = 1, document in category
            # if all five category binaries == 0, then categorize as 'unkn'
            # for cat in cats
            #     # target = [[label1], [label1], [label2], [label3], [label3],...[labeln]]
            # ... target.append([cat])
            # ... data.append([text])


    target = np.array(target)

    if encoding is not None:
            data = [d.decode(encoding, decode_error) for d in data]

    # # label is index of folder (0-19)
    # for label, folder in enumerate(folders):
    #     target_names.append(folder)
    #     folder_path = join(container_path, folder)
    #     documents = [join(folder_path, d)
    #                  for d in sorted(listdir(folder_path))]

    #     target.extend(len(documents) * [label])
    #     filenames.extend(documents)

    # # convert to array for fancy indexing
    # filenames = np.array(filenames)
    # target = np.array(target)

    if load_content:
        data = []
        for filename in filenames:
            with open(filename, 'rb') as f:
                data.append(f.read())

    return Bunch(data=data,
                 filenames=filenames,
                 target_names=target_names,
                 target=target,
                 DESCR=description)
