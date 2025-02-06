import functions_framework
import json
import numpy as np
from openai import OpenAI
import os
import pickle
import sklearn
from google.cloud import storage

api_key = os.environ.get('api_key')
openai_client = OpenAI(api_key=api_key)

storage_client = storage.Client()
bucket = storage_client.bucket(os.environ.get('bucket-name'))

tse_blob = bucket.blob('title-summary-embedding-256.json')
temp_file = '/tmp/title-summary-embedding.json'
tse_blob.download_to_filename(temp_file)
with open(temp_file,'r') as tf:
    title_summary_embedding = json.load(tf)
all_titles = list(title_summary_embedding.keys())
all_values = list(title_summary_embedding.values())
all_embeddings = [value[1] for value in all_values]

pca_blob = bucket.blob('pca.pkl')
pca = pickle.loads(pca_blob.download_as_bytes())

model = 'gpt-3.5-turbo-0125'
def generate_book_summary(title,author):

    system_content = """You are a book summarizer specializing in creating concise and clear 
summaries of books for a recommendation system, which will work by comparing the vector 
embeddings of summaries using dot product. Your goal is to provide a brief summary of a given 
book, highlighting the plot, themes, appeal, genre, and whatever else may stand out. Be engaging 
and avoid spoilers.""".replace('\n','')

    author = f' by {author}' if author else ''

    prompt = f"""Provide a brief summary of the book "{title}"{author}. Briefly highlight 
the books's plot, themes, appeal, etc. If the book is non-fiction, explain what it intends to 
do.""".replace('\n','')

    messages = [
        {'role':'system',
        'content':system_content},
        {'role':'user',
        'content':prompt}
    ]

    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1.2
    )

    summary = response.choices[0].message.content
    
    return summary

embedding_model = 'text-embedding-3-small'
generate_embedding = lambda summary: openai_client.embeddings.create(
    input=summary,model=embedding_model
    ).data[0].embedding

def get_similar_books(target_embedding,num_books=6):

    all_dot_products = {
        all_titles[i]:np.dot(target_embedding,all_embeddings[i]) 
        for i in range(len(all_titles))
    }
    ordered_dot_products = sorted(all_dot_products.items(),key=lambda i: i[1],reverse=True)
    top_titles = [p[0] for p in ordered_dot_products[:num_books]]
    top_summaries = [title_summary_embedding[t][0] for t in top_titles]

    return top_titles,top_summaries


@functions_framework.http
def generate_recommendations(request):
    
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type"
    }

    if request.method == "OPTIONS":
        return ('', 204, headers)

    request_json = request.get_json(silent=True)

    if request_json and 'title' in request_json:
        title = request_json['title']
        title = title[:220]
    else:
        title = None
        return title
    if request_json and 'author' in request_json:
        author = request_json['author']
        author = author[:220]
    else:
        author = None
    
    summary = generate_book_summary(title,author)
    embedding = generate_embedding(summary)
    reduced_embedding = pca.transform(np.expand_dims(embedding,0)).squeeze()

    top_titles,top_summaries = get_similar_books(reduced_embedding)

    response_data = {
        "top_titles":top_titles,
        "top_summaries":top_summaries
    }
        
    return (response_data,200,headers)