# Movie Recommendation System

# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 2: Load Dataset (MovieLens small dataset assumed)
# Download: https://grouplens.org/datasets/movielens/
movies = pd.read_csv('D:\movierecomendation\movieenv\data_set\movies.csv')
ratings = pd.read_csv(r'D:\movierecomendation\movieenv\data_set\ratings.csv')

print("Movies:", movies.shape)
print("Ratings:", ratings.shape)

# Step 3: Collaborative Filtering (User-Item Matrix)
user_item_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')
user_item_matrix.fillna(0, inplace=True)

# Compute cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Function: Get collaborative filtering recommendations
def recommend_cf(user_id, num_recommendations=5):
    # Find similar users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]
    
    # Aggregate ratings from similar users
    similar_user_ratings = user_item_matrix.loc[similar_users]
    weighted_ratings = similar_user_ratings.T.dot(user_similarity_df[user_id][similar_users])
    weighted_ratings /= user_similarity_df[user_id][similar_users].sum()
    
    # Remove movies already rated by target user
    user_rated_movies = user_item_matrix.loc[user_id]
    recommendations = weighted_ratings[user_rated_movies[user_rated_movies == 0].index]
    
    # Top-N recommendations
    top_movies = recommendations.sort_values(ascending=False).head(num_recommendations)
    return movies[movies['movieId'].isin(top_movies.index)][['movieId','title']]

# Step 4: Content-Based Filtering (Based on genres)
# Preprocess genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'].fillna(''))

# Compute cosine similarity between movies
content_similarity = cosine_similarity(tfidf_matrix)

# Function: Recommend based on a movie (content-based)
def recommend_content(movie_title, num_recommendations=5):
    idx = movies[movies['title'].str.lower() == movie_title.lower()].index[0]
    sim_scores = list(enumerate(content_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][['movieId','title']]

# Step 5: Example Usage
print("Collaborative Filtering Recommendations for User 1:")
print(recommend_cf(user_id=1))

print("\nContent-Based Recommendations for 'Toy Story (1995)':")
print(recommend_content('Toy Story (1995)'))
