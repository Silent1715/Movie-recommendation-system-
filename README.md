# Movie Recommendation System

A simple movie recommendation system using the [MovieLens dataset](https://grouplens.org/datasets/movielens/).  
It includes two approaches: **Content-Based Filtering** and **Collaborative Filtering**.

## Features
- Content-Based: recommends movies based on genres using TF-IDF and cosine similarity.  
- Collaborative: recommends movies for a user based on similar users’ ratings.  
- Outputs top 5 recommendations.

## Dataset
Download the MovieLens dataset and place `movies.csv` and `ratings.csv` in the project folder.  

## Installation
```bash
pip install pandas numpy scikit-learn
