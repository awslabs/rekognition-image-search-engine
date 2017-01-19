'''
Local Image Search Engine

@Author: Sunil Mallya
'''
import numpy as np
import pickle
import json
import sys

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity 

# Load Index
idx_file = 'myphotoindex.dvect'
d_index = pickle.load(open(idx_file, 'r'))
indices = np.array(d_index.keys())

# Load Model
tfidf = pickle.load(open('tfidf_model.pkl'))

# Load features
features = pickle.load(open('features.pkl'))

### WEB ### 
from flask import Flask
from flask import request
from flask.ext.cors import CORS, cross_origin
app = Flask(__name__)

cors = CORS(app, resources={r"/*": {"origins": "*"}})

# stop words
stop_words = [u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your',
        u'yours', u'yourself', u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her',
        u'hers', u'herself', u'it', u'its', u'itself', u'they', u'them', u'their', u'theirs',
        u'themselves', u'what', u'which', u'who', u'whom', u'this', u'that', u'these', u'those',
        u'am', u'is', u'are', u'was', u'were', u'be', u'been', u'being', u'have', u'has', u'had',
        u'having', u'do', u'does', u'did', u'doing', u'a', u'an', u'the', u'and', u'but', u'if',
        u'or', u'because', u'as', u'until', u'while', u'of', u'at', u'by', u'for', u'with',
        u'about', u'against', u'between', u'into', u'through', u'during', u'before', u'after',
        u'above', u'below', u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over',
        u'under', u'again', u'further', u'then', u'once', u'here', u'there', u'when', u'where',
        u'why', u'how', u'all', u'any', u'both', u'each', u'few', u'more', u'most', u'other',
        u'some', u'such', u'no', u'nor', u'not', u'only', u'own', u'same', u'so', u'than', u'too',
        u'very', u's', u't', u'can', u'will', u'just', u'don', u'should', u'now', u'd', u'll', u'm',
        u'o', u're', u've', u'y', u'ain', u'aren', u'couldn', u'didn', u'doesn', u'hadn', u'hasn',
        u'haven', u'isn', u'ma', u'mightn', u'mustn', u'needn', u'shan', u'shouldn', u'wasn',
        u'weren', u'won', u'wouldn', 'people', 'image', 'images']

########## APP ENDPOINTS ################################

@app.route('/search')
@cross_origin(origin='localhost', headers=['Content- Type','Authorization'])
def search():
    qry = request.args.get('query', '')
    test = np.zeros((tfidf[0].shape))

    keywords = []

    for word in qry.split(' '):
        # validate word
        if len(word) <2 or word in stop_words:
            continue 
        try:
            idx = features.index(word)
            test[0][idx] = 1
        except ValueError, e:
            pass

    cosine_similarities = cosine_similarity(test, tfidf).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-100:-1] # TOP 100 results
  
    MAX = 100 
    data = []
    related_docs_indices = related_docs_indices[:MAX]
    tag_map = {} # All tags and their counts
    
    for img in indices[related_docs_indices]:
        file_path = "/Users/smallya/workspace/Rekognition-personal-searchengine/" + img
        labels = d_index[img]
        word = qry.split(' ')[0]
        data.append(file_path)

    print related_docs_indices
    return json.dumps(data) 

if __name__ == "__main__":
    app.run(port=8000, threaded=True)
