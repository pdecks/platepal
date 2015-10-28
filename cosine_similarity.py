"""
Documentation and comments by Patricia Decker. 2015-10-27

Basic Statistical NLP Part 2 - TF-IDF and Cosine Similarity
by: Bill Chambers, 2014-12-22
http://billchambers.me/tutorials/2014/12/22/cosine-similarity-explained-in-python.html
"""

from __future__ import division
import string
import math

# tokenize words on spaces
tokenize = lambda doc: doc.lower().split(" ")

# sample documents
document_0 = "China has a strong economy that is growing at a rapid pace. However politically it differs greatly from the US Economy."
document_1 = "At last, China seems serious about confronting an endemic problem: domestic violence and corruption."
document_2 = "Japan's prime minister, Shinzo Abe, is working towards healing the economic turmoil in his own country for his view on the future of his people."
document_3 = "Vladimir Putin is working hard to fix the economy in Russia as the Ruble has tumbled."
document_4 = "What's the future of Abenomics? We asked Shinzo Abe for his views"
document_5 = "Obama has eased sanctions on Cuba while accelerating those against the Russian Economy, even as the Ruble's value falls almost daily."
document_6 = "Vladimir Putin is riding a horse while hunting deer. Vladimir Putin always seems so serious about things - even riding horses. Is he crazy?"

# list of all documents
all_documents = [document_0, document_1, document_2, document_3, document_4, document_5, document_6]

# tokenize the documents
# tokenized_document_list = [tokenize(d) for d in all_documents]

# all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
# equivalent to ..
# tokens_list = []
# for sublist in tokenized_document_list:
#       for item in sublist:
#           tokens_list.append(item)
# set(tokens_list)
# check:
# all_tokens_set == set(tokens_list)
# >>> True


def jaccard_similarity(query, document):
    """Returns the Jaccard Similarity of two documents.

    Query = set of tokens for document 1 (set A)
    Document = set of tokens for document 2 (set B)

    J(A, B) = length(intersection of two sets) / length (union of two sets)


    Test that the jaccard_similarity of a document with itself is equal to 1.0:

    >>> jaccard_similarity(tokenized_document_list[2],tokenized_document_list[2])
    >>> 1.0

    """
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)


def term_frequency(term, tokenized_document):
    """The count (frequency) of a term in a document."""
    return tokenized_document.count(term)


def sublinear_term_frequency(term, tokenized_document):
    """Common term frequency modification that uses the logarithm of the term frequency.

    Why? It seems unlikely that twenty occurences of a term in a document truly
    carry 20x the signifance of a single occurrence.

    source: http://nlp.stanford.edu/IR-book/html/htmledition/sublinear-tf-scaling-1.html
    """
    # count = tokenized_document.count(term)  # replaced with line below
    count = term_frequency(term, tokenized_document)
    if count == 0:
        return 0
    return 1 + math.log(count)


def augmented_term_frequency(term, tokenized_document):
    """Returns raw frequency divided by the maximum raw frequency of any term
    in the document. Double normalization with K = 0.5."""
    max_count = max([term_frequency(t, tokenized_document) for t in tokenized_document])
    return (0.5 + ((0.5 * term_frequency(term, tokenized_document))/max_count))


def inverse_document_frequencies(tokenized_documents):
    """Returns a dictionary where key = token and value = idf(t) 

    Inverse doc. freq. targets the words that are unique to certain documents.
    The idf of a rare term is high, whereas the idf of a frequent term is
    likely to be low.

    For every token, t, in the set of all tokens for all documents, D, with
    a token appearing in a particular document, d:

    idf(t, D) = log[ N / |d in D : t in d| ] = log [N / df(t)]
    where dft = document frequency of a token, t

    see also:
    http://nlp.stanford.edu/IR-book/html/htmledition/inverse-document-frequency-1.html
    """
    idf_values = {}

    # take the list of lists (where each sublist contains non-unique tokens
    # for a single document in documents list), return a set of unique tokens
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    
    # for each unique token, computer its idf
    for tkn in all_tokens_set:
        # remember: map(func, seq), so here the sequence is tokenized_documents
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
    
    return idf_values


def tfidf(documents):
    """Takes a list of documents and returns a list of lists.

    Transforms documents into numbers and lists of documents into matrices.
    tf-idf weighting uses a composite weight for each term in each document.
    Assigns to term t a weight in document d given by:

    tf-idf(t,d) = tf(t,d) x idf(t)

    tf-idf(t) is:
    1. highest when t occurs many times within a small number of documents
       (thus lending high discriminating power to those documents);
    2. lower when the term occurs fewer times in a document, or occurs in many
       documents (thus offering a less pronounced relevance signal);
    3. lowest when the term occurs in virtually all documents.

    see also:
    http://nlp.stanford.edu/IR-book/html/htmledition/tf-idf-weighting-1.html
    """
    # tokenize each document in list of documents
    #   returns a list of lists, where each list contains
    #   non-unique tokens for a single document in documents
    tokenized_documents = [tokenize(d) for d in documents]

    # calculate the idf of each unique token
    idf = inverse_document_frequencies(tokenized_documents)
    tfidf_documents = []
    for document in tokenized_documents:
        doc_tfidf = []
        for term in idf.keys():
            tf = sublinear_term_frequency(term, document)
            doc_tfidf.append(tf * idf[term])
        tfidf_documents.append(doc_tfidf)

    return tfidf_documents

# added alternate version of tfidf (pd)
def tfidf_dict(documents):
    """Alternate version of tfidf that returns list of dictionaries for
    visualizing tfidf terms.
    """
    tokenized_documents = [tokenize(d) for d in documents]

    idf = inverse_document_frequencies(tokenized_documents)
    tfidf_dict = []
    for document in tokenized_documents:
        doc_tfidf_dict = {}
        for term in idf.keys():
            tf = sublinear_term_frequency(term, document)
            tf_idf = tf * idf[term]
            doc_tfidf_dict[term] = tf_idf
        tfidf_dict.append(doc_tfidf_dict)
    return tfidf_dict

#in Scikit-Learn
from sklearn.feature_extraction.text import TfidfVectorizer

sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
sklearn_representation = sklearn_tfidf.fit_transform(all_documents)


########### END BLOG POST 1 #############

def cosine_similarity(vector1, vector2):
    """Returns scalar cosine similarity, a measure of the similarity between two
    (document) vectors, A and B.

    similarity = cos(theta) = ( A dot B ) / ((len A) * (len B))

    If cosine similarity == 1, the documents are the same document.
    If '' '' == 0, the documents share nothing in common.
    (term frequency cannot be negative, so angle between vectors cannot
     exceed 90 degrees)
    """
    dot_product = sum(p*q for p,q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
    if not magnitude:
        return 0
    return dot_product/magnitude


tfidf_representation = tfidf(all_documents)
our_tfidf_comparisons = []
for count_0, doc_0 in enumerate(tfidf_representation):
    for count_1, doc_1 in enumerate(tfidf_representation):
        our_tfidf_comparisons.append((cosine_similarity(doc_0, doc_1), count_0, count_1))

skl_tfidf_comparisons = []
for count_0, doc_0 in enumerate(sklearn_representation.toarray()):
    for count_1, doc_1 in enumerate(sklearn_representation.toarray()):
        skl_tfidf_comparisons.append((cosine_similarity(doc_0, doc_1), count_0, count_1))

print "each tuple: (cosine_similarity(doc_0, doc_1), count outer, count inner)"
print "first tuple: our_tfidf_comparisons  |  second tuple: skl_tfidf_comparisons"
for x in zip(sorted(our_tfidf_comparisons, reverse=True), sorted(skl_tfidf_comparisons, reverse=True)):
    print x

## Reprint for clarity ... ##
print "our_tfidf_comparisons:"
for item in our_tfidf_comparisons:
    print item

print "skl_tfidf_comparisons"
for item in skl_tfidf_comparisons:
    print item

## Added tfidf representation that is easier to read / understand, for learning purposes only ##
pd_tfidf = []
# outer loop
print "cosine similarity (A, B) | vector A | vector B"
for i in range(len(tfidf_representation)):
    # inner loop
    for j in range(len(tfidf_representation)):
        pd_tuple = (cosine_similarity(tfidf_representation[i], tfidf_representation[j]), "document_"+str(i), "document_"+str(j))
        pd_tfidf.append(pd_tuple)
## END pd _tfidf ##
