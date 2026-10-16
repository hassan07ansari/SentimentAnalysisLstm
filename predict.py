import numpy as np
from gensim.models import Word2Vec
from tensorflow.keras.models import load_model


from preprocessing import preprocess

word2vec_model = Word2Vec.load("word2vec.model")
lstm_model = load_model("model_lstm.keras")

MAX_LENGTH = 309
VECTOR_SIZE = 100

def predict_sentiment(review):
# Step 1: Preprocess the review
    words = preprocess(review)

# Step 2: Convert words into Word2Vec vectors
    vectors = [
        word2vec_model.wv[word]
        for word in words
        if word in word2vec_model.wv
]

# Step 3: Create a padded input
    padded = np.zeros(
        (1, MAX_LENGTH, VECTOR_SIZE),
        dtype=np.float32
)

    length = min(len(vectors), MAX_LENGTH)

    if length > 0:
        padded[0, :length] = np.asarray(
            vectors[:length],
            dtype=np.float32
        )

# Step 4: Predict sentiment
    probability = float(
        lstm_model.predict(padded, verbose=0)[0][0]
    )

# Step 5: Convert probability to label
    sentiment = "Positive" if probability >= 0.5 else "Negative"

    return {
        "sentiment": sentiment,
        "positive_probability": round(probability, 4)
    }


if __name__ == "__main__":
    review = "The movie was fantastic. I really enjoyed it!"
    print(predict_sentiment(review))

