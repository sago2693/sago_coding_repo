from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

app = FastAPI()

# Define a reference dictionary of lemmas for each word
reference_lemmas = {
    "behaviour": "behave",
    "donation": "donate",
    "sparkingr": "spark"
}

# Define a Pydantic model for the input text
class InputText(BaseModel):
    text: str

# Define a route for the root endpoint of the API
@app.get("/")
async def root():
    """
    A simple endpoint that returns a message indicating the API is working.
    """
    return {"message": "API working"}

# Define a route for the lemmatize endpoint of the API
@app.get("/lemmatize/")
async def preprocess(request: InputText):
    """
    An endpoint that tokenizes the input text and returns a dictionary of lemmas
    for each token based on a reference dictionary of lemmas.
    
    Parameters:
    - request (InputText): A Pydantic model representing the input text to be lemmatized.
    
    Returns:
    - lemmas (Dict[str, str]): A dictionary where each key is a token from the input text
      and the corresponding value is the lemma for that token according to the reference dictionary.
      If a token is not found in the reference dictionary, the value is "not_found".
    """
    # Get the text from the request
    text = str(request.text)

    # Tokenize the text using NLTK
    tokens = word_tokenize(text)

    # Look up the lemmas for each token in the reference dictionary
    lemmas = {}
    for token in tokens:
        lemma = reference_lemmas.get(token, "not_found")
        lemmas[token] = lemma

    # Return the dictionary of lemmas
    return lemmas
