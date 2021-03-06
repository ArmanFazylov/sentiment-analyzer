## Text Sentiment Analysis with Keras CNN and Word2Vec

This project is mostly a mix of following two projects [p1](https://vgpena.github.io/classifying-tweets-with-keras-and-tensorflow/) and [p2](https://github.com/giuseppebonaccorso/twitter_sentiment_analysis_word2vec_convnet).

Many thanks guys! :)

Data can be downloaded [here](http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip).

### Installation

You need Python 2.7 to run this project.

Run `pip install -r requirements.txt` 

Note: if want to use dictionary stemmers like Hunspell - make sure to pre-install `python-dev` and `libhunspell-dev` on your machine and folder `/usr/share/hunspell` with dicts exists.
However, snowball stemmers does the job well.

Run `python -m nltk.downloader stopwords`

Note: in case you get this error ` ImportError: libcublas.so.8.0: cannot open shared object file: No such file or directory`:

You should:
 1. Install latest nvidia drivers

 2. Install latest CUDA
            
 3. Compile and install tensorflow from source. Tutorial [here](http://www.python36.com/install-tensorflow141-gpu/#comment-573)

### Training

Then run `run train.py` ( process takes considerable amount of time)

These new files should be created: `word2vec`, `model.json`, and `model.h5`. 

### Classification

To analyse sentiment of user input, run `python play.py` and type into the console when prompted. 

Hitting `Enter` without typing anything will quit the program.

![gif](https://github.com/ArmanFazylov/text_sentiment_analyzer/blob/master/play.gif)