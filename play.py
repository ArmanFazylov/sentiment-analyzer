import json
import numpy as np
from keras.models import model_from_json
import keras.backend as K
import multiprocessing
import tensorflow as tf
import numpy as np
from gensim.models.word2vec import Word2Vec
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import os

# Clear nvidia
os.system('rm -rf ~.nv/')

model_location = 'model/'

# Select whether using Keras with or without GPU support
# See: https://stackoverflow.com/questions/40690598/can-keras-with-tensorflow-backend-be-forced-to-use-cpu-or-gpu-at-will
use_gpu = True

config = tf.ConfigProto(intra_op_parallelism_threads=multiprocessing.cpu_count(),
                        inter_op_parallelism_threads=multiprocessing.cpu_count(),
                        allow_soft_placement=True,
                        device_count={'CPU': 1,
                                      'GPU': 1 if use_gpu else 0})

session = tf.Session(config=config)
K.set_session(session)

# Load word2vec
word2vec = Word2Vec.load(model_location + 'word2vec')
X_vecs = word2vec.wv

# Test word2vec
print('testin word2vec...')
print(X_vecs.most_similar(positive=['man', 'gold'], negative=['money']))

del word2vec

# Init Tokenizer and stemmers
tkr = RegexpTokenizer('[a-zA-Z0-9@]+')
stemmer = SnowballStemmer('english')
# Get the list of stop words
stop_words = stopwords.words('english')

# Read in your saved model structure
json_file = open(model_location + 'model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# and create a model from that
model = model_from_json(loaded_model_json)

# and weight your nodes with your saved values
model.load_weights(model_location + 'model.h5')

max_tweet_length = 15
vector_size = 512

# Interactive part
while 1:
    evalSentence = raw_input('Input a sentence to be evaluated, or Enter to quit: ')
    if len(evalSentence) == 0:
        break
    # Tokenize the sentence
    tokenized_corpus = []

    # Tokenize the tweet
    tokens = tkr.tokenize(evalSentence)
    # Remove the stop words
    #tokens = [t for t in tokens if not t in stop_words]
    # Stem
    tokens = [stemmer.stem(t) for t in tokens if not t.startswith('@')]

    tokenized_corpus.append(tokens)

    print('tokenized_corpus length is ' + str(len(tokenized_corpus)))
    print(tokenized_corpus)

    # prepare empty array
    X = np.zeros((1, max_tweet_length, vector_size), dtype=K.floatx())

    for t, token in enumerate(tokenized_corpus[0]):
        if t >= max_tweet_length:
            break
        if token not in X_vecs:
            print('skipping token ' + token)
            continue
        # fill prediction vector
        X[0, t, :] = X_vecs[token]

    # for human-friendly printing
    labels = ['negative', 'positive']

    # Predict
    prediction = model.predict(X)

    print(prediction)

    print("%s sentiment; %f%% confidence" % (labels[np.argmax(prediction)], prediction[0][np.argmax(prediction)] * 100))
