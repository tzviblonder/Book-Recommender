# Book Recommender System
## Overview

#### This is a book recommendation system that suggests similar books based on user input. It uses OpenAI embeddings and PCA for dimensionality reduction, comparing book summaries to a vector database of 120,000+ books.
<ul>
    <li>Frontend: HTML/CSS & JavaScript, hosted on Netlify</li>
    <li>Backend: Python API, hosted as a Google Cloud Function on GCP</li>
    <li>Database: 120,000+ book titles, summaries, and 256-dimensional embeddings</li>
</ul>

#### How It Works:
1. The users enters a book title and author.
2. This information is sent to the backend, which generates a summary of that book using OpenAI's API.
3. The summary is converted into a 1536-dimensional vector embedding using OpenAI's embedding model.
4. The embedding is reduced using PCA (principal component analysis) into 256 components to reduce memory and computation.
5. The system finds similar books by computing the dot product of the new embedding with stored embeddings.
6. A list of recommended books and their summaries is returned.
