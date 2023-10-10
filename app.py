#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2023 * Ltd. All rights reserved.
#
#   Editor      : Pycharm
#   File name   : app.py
#   Author      : Akash Arya
#   Created date: 2023-10-10 12:30:26
#   Description : This code sets up a semantic search API using the Flask framework, which allows users to find similar food items based on their input query.
#
#================================================================


# IMPORT
from flask import Flask, request, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import requests
import numpy as np

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('calories.csv')  # Downloaded the dataset

# Load the SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Create an index using FAISS
fooditem_embeddings = model.encode(df['FoodItem'].tolist(), convert_to_tensor=True)
index = faiss.IndexFlatL2(fooditem_embeddings.shape[1])
index.add(fooditem_embeddings.numpy())

# Define an API endpoint for semantic search
@app.route('/',methods=['GET','POST'])
def semantic_searchX():
    print('Enter semantic_search')
    # Encode the query
    query_embedding = model.encode("Blackberries", convert_to_tensor=True)
    # print(query_embedding.numpy())
    # print(query_embedding.shape)

    xt = query_embedding.numpy()
    xt = np.expand_dims(xt,axis=0)
    # print(xt.shape)

    # Search for similar food items
    k = 20  # Number of similar items to retrieve
    # distances, indices = index.search(query_embedding.numpy(), k)
    distances, indices = index.search(xt, k)

    # Get the top-k similar food items
    similar_food_items = df.iloc[indices[0]]['FoodItem'].tolist()
    print("*"*10)
    print(similar_food_items)
    return jsonify({"results": similar_food_items})

if __name__ == '__main__':
    app.run()

