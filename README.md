# IMDb Sentiment Analysis using LSTM

A Natural Language Processing (NLP) project that uses Word2Vec embeddings and a Long Short-Term Memory (LSTM) neural network to classify IMDb movie reviews as positive or negative.

## Dataset

The project uses 50,000 IMDb movie reviews:

* Positive reviews: 25,000
* Negative reviews: 25,000

The dataset is balanced, with equal numbers of positive and negative reviews.

## Data Preprocessing

Text preprocessing and Word2Vec embeddings are used to prepare the reviews for the LSTM model.

Stratified splitting is used to preserve the positive-to-negative class distribution across the training, validation, and test sets.

## Model

* Word2Vec for word embeddings
* LSTM neural network for sentiment classification
* Binary classification: positive or negative

## Project Files

* `IMDB.ipynb` — IMDb dataset preparation and preprocessing.
* `model_lstm.keras` — Saved LSTM model.
* `word2vec.model` — Trained Word2Vec model.
* `word2vec.model.syn1neg.npy` — Word2Vec model data.
* `word2vec.model.wv.vectors.npy` — Word vectors.

## Future Improvements

* Evaluate model performance using appropriate classification metrics.
* Build an interactive application for sentiment prediction.
* Deploy the application for public use.
