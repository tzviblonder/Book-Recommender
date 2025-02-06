# Book Recommender System
## Overview

####This is a book recommendation system that suggests similar books based on user input. It uses OpenAI embeddings and PCA for dimensionality reduction, comparing book summaries to a vector database of 120,000+ books.

    Frontend: HTML, CSS, JavaScript (hosted on Netlify)
    Backend: Python API (hosted as a Google Cloud Function)
    Database: 1536-dimensional OpenAI embeddings reduced using PCA

#### How It Works

    The user enters a book title and author.
    The backend retrieves or generates a summary using OpenAI’s API.
    The summary is converted into a vector embedding and reduced using PCA.
    The system finds similar books by computing the dot product with stored embeddings.
    A list of recommended books is returned.
