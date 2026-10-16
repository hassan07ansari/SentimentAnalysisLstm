import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# Load English stopwords once
STOP_WORDS = set(stopwords.words("english"))


def preprocess(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = ''.join(
        char for char in text
        if char not in string.punctuation
    )

    # Tokenize the text
    words = word_tokenize(text)

    # Remove English stopwords
    words = [word for word in words if word not in STOP_WORDS]

    return words




#testing

import nltk

nltk.download("punkt_tab")
nltk.download("stopwords")

from preprocessing import preprocess

review = "The movie was AMAZING! I loved it."
print(preprocess(review))



if __name__ == "__main__":
    review = "The movie was AMAZING! I loved it."
print(preprocess(review))