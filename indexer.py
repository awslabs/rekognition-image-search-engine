'''
Local Image Search Engine Indexer

@Author: Sunil Mallya
'''
import boto3
import glob
import json
import pickle

from multiprocessing.dummy import Pool as ThreadPool
from threading import Lock

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# PHOTOS Directory
photo_dir = 'photos/'

# Rekognition client
rekognition = boto3.client('rekognition', region_name='us-east-1', 
        endpoint_url='https://rekognition.us-east-1.amazonaws.com')

# Document Index: Lets create a "image document" vector
d_index = {} # doc => [{l,c}....] 
lock = Lock()

# List of images to index
files = glob.glob(photo_dir + "*.jpg")
files.extend(glob.glob(photo_dir + "*.JPG"))

# Make calls to Rekognition
label_counts = {}

## Extract features helper
def get_labels(f):
    try:
        with open(f, 'rb') as image:
            resp = rekognition.detect_labels(Image={'Bytes': image.read()})

        # TODO: if person/human found, then get face features as well
        # TODO: Index exif data

        labels = resp['Labels']
        
        dt = {}
        for v in labels:
            l = v['Name'].lower() 
            c = v['Confidence']
            
            # Choose an appopriate confidence level 
            # based on your application
            if c < 70:
                continue
            
            # insert features
            dt[l] = c
            
            # count label freq for word cloud generation
            try:
                label_counts[l]
            except KeyError:
                label_counts[l] = 0
            label_counts[l] += 1

        lock.acquire()
        d_index[f] = dt
        lock.release()
        

    except Exception, e:
        print e
    finally:
        pass #lock.release()

# Use ThreadPool to make concurrent requests 
N_THREADS = 25 
pool = ThreadPool(N_THREADS)
results = pool.map(get_labels, files)
pool.close()
pool.join()

### Lets save the data ###
data = pickle.dumps(d_index)
with open('myphotoindex.dvect', 'w') as f:
    f.write(data)

# Generate Search Index/ TF-IDF Model
vec = DictVectorizer()
counts = vec.fit_transform(d_index.values()).toarray()
transformer = TfidfTransformer(smooth_idf=True)
tfidf = transformer.fit_transform(counts)

# TF_IDF Model
vals = pickle.dumps(tfidf)
with open('tfidf_model.pkl', 'w') as f:
    f.write(vals)

# Features available 
features = vec.get_feature_names()
vals = pickle.dumps(features)
with open('features.pkl', 'w') as f:
    f.write(vals)


### Generate word tag cloud 

from wordcloud import WordCloud
wordcloud = WordCloud()

from operator import itemgetter
item1 = itemgetter(1)
frequencies = sorted(label_counts.items(), key=item1, reverse=True)
wordcloud.generate_from_frequencies(frequencies)

# save image
import matplotlib.pyplot as plt
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('web/photo_tags')
