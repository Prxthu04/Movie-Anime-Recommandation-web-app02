import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate

# 1. Load Combined Dataset
df = pd.read_csv("data/combined_ratings.csv")

# 2. Encode User & Item IDs
df["user_idx"] = df["user_id"].astype("category").cat.codes
df["item_idx"] = df["item_id"].astype("category").cat.codes

n_users = df["user_idx"].nunique()
n_items = df["item_idx"].nunique()

# 3. Train/Test Split
train, test = train_test_split(df, test_size=0.2, random_state=42)

# 4. Define Light Neural Net Model
embedding_size = 50

user_input = Input(shape=(1,))
item_input = Input(shape=(1,))

user_embedding = Embedding(n_users, embedding_size)(user_input)
item_embedding = Embedding(n_items, embedding_size)(item_input)

user_vec = Flatten()(user_embedding)
item_vec = Flatten()(item_embedding)

# Optional: add some dense layers (deep part)
concat = Concatenate()([user_vec, item_vec])
dense = Dense(64, activation='relu')(concat)
output = Dense(1)(dense)

model = Model([user_input, item_input], output)
model.compile(optimizer='adam', loss='mse')

# 5. Train the model
model.fit(
    [train.user_idx, train.item_idx],
    train.rating,
    epochs=5,
    batch_size=256,
    validation_split=0.1
)

# 6. Save model & mappings
model.save("model/hybrid_model.h5")
df[["user_id", "user_idx"]].drop_duplicates().to_csv("model/user_map.csv", index=False)
df[["item_id", "item_idx", "title", "type", "genre", "source"]].drop_duplicates().to_csv("model/item_map.csv", index=False)

print("✅ Model trained and saved.")
