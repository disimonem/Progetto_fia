''' 
#def select_by_importance(self, threshold=0.01):
 #   """Seleziona le caratteristiche basate sull'importanza con RandomForest."""
threshold = 0.01
model = RandomForestClassifier()
model.fit(X, y)
importance = model.feature_importances_
selected_features = X.columns[importance > threshold]
# Plot the forest of trees
plt.figure(figsize=(20, 10))
plot_tree(model.estimators_[0], filled=True, rounded=True, class_names=True, feature_names=selected_features)
plt.title("Random Forest Tree")
plt.show()
''' 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
import feature_selection 

# Load and preprocess the dataset
dataset = pd.read_parquet('challenge_campus_biomedico_2024.parquet')
df = feature_selection.dataset_preprocessing(dataset)
print(df.head())

# Split the dataset into features and labels
X = df.drop(['label'], axis=1)
y = df['label']

# Define threshold for feature selection based on importance
threshold = 0.01

# Train a RandomForest model
model = RandomForestClassifier()
model.fit(X, y)

# Get feature importances
importance = model.feature_importances_
selected_features = X.columns[importance > threshold]

# Check how many trees are in the forest
num_trees = len(model.estimators_)
print(f"Number of trees in the forest: {num_trees}")

# Plot the 6th tree in the forest (index 5)
tree_index_to_plot = 5

# Ensure selected_features is the list of all features that were used to train
plt.figure(figsize=(20, 10))
plot_tree(
    model.estimators_[tree_index_to_plot], 
    filled=True, 
    rounded=True, 
    class_names=np.unique(y).astype(str),  # Ensure proper class names
    feature_names=X.columns  # Use all features since RandomForest uses all features for each tree
)
plt.title(f"Random Forest Tree {tree_index_to_plot}")
plt.show()
