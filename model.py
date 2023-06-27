import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset containing customer purchase history
dataset = pd.read_csv('customer_purchase_history.csv')

# Perform data preprocessing and feature engineering
# ...

# Create a user-item matrix
user_item_matrix = dataset.pivot_table(index='CustomerID', columns='ProductID', values='PurchaseCount')

# Calculate item-item similarity matrix using cosine similarity
item_similarity = cosine_similarity(user_item_matrix.fillna(0))

# Function to generate personalized recommendations for a given user
def generate_recommendations(user_id, top_n):
    user_ratings = user_item_matrix.loc[user_id]
    similar_items = pd.Series(0, index=user_item_matrix.columns)
    
    # Calculate the weighted average of item ratings based on similarity scores
    for item_id, rating in user_ratings.iteritems():
        similar_items += item_similarity[item_id] * rating
    
    # Exclude items already purchased by the user
    similar_items = similar_items.drop(user_ratings.index)
    
    # Sort items based on their weighted ratings
    top_items = similar_items.sort_values(ascending=False).head(top_n)
    
    return top_items.index.tolist()

# Generate personalized recommendations for a specific user
user_id = '12345'
top_n = 5
recommendations = generate_recommendations(user_id, top_n)

# Print the recommended product IDs
print(f"Recommended Products for User {user_id}:")
for product_id in recommendations:
    print(product_id)
