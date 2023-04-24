Sure, here's an example of a `README.md` file that explains how to run the API:

# Lemmatization API

This is a simple API that lemmatizes text using a reference dictionary of lemmas. It is built using FastAPI and NLTK.

## Installation

1. Clone this repository to your local machine.
2. Install the required packages by running `pip install -r requirements.txt`.
3. Download the NLTK punkt tokenizer by running the following code in Python:

```python
import nltk
nltk.download('punkt')
```

## Usage

To start the API server, run the following command in your terminal:

```sh
uvicorn main:app --reload
```

This will start the API server at `http://localhost:8000`. You can then send requests to the API using a tool like `curl` or a web-based API client like Postman.

### Endpoints

The API has the following endpoints:

#### `/`

This endpoint returns a simple message indicating that the API is working.

Example usage:

```sh
curl http://localhost:8000/
```

Example response:

```json
{
  "message": "API working"
}
```

#### `/lemmatize`

This endpoint lemmatizes the input text using a reference dictionary of lemmas.

Example usage:

```sh
curl -X GET -H "Content-Type: application/json" -d '{"text":"behaviour donation sparking"}' http://localhost:8000/lemmatize/
```

Example response:

```json
{
  "behaviour": "behave",
  "donation": "donate",
  "sparking": "spark"
}
```

### API documentation

The API documentation is automatically generated and can be accessed at `http://localhost:8000/docs` or `http://localhost:8000/redoc`.

## Contributing

If you find any bugs or have suggestions for how to improve this API, please feel free to open an issue or submit a pull request.

## License

This API is released under the MIT License. See `LICENSE` for more information.