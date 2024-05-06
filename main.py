import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# Load the dataset from a CSV file
file_path = 'Dataset/Generate_synthetic_fraud_data.csv'
data = pd.read_csv(file_path)

# Extracting features (X) from the dataset
X_train = data.drop(columns=['transaction_id', 'timestamp', 'amount', 'merchant_id', 'customer_id'])  # Remove the 'is_fraud' column to get the features

# Extracting the target variable (y) from the dataset
y_train = data['is_fraud']  # 'is_fraud' column contains the labels

# Initialize the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the training data
y_train_pred = clf.predict(X_train)

# Calculate performance metrics on the training data
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1_score = f1_score(y_train, y_train_pred)
train_roc_auc = roc_auc_score(y_train, y_train_pred)

# Print the performance metrics on the training data
print("Training Accuracy:", train_accuracy)
print("Training Precision:", train_precision)
print("Training Recall:", train_recall)
print("Training F1 Score:", train_f1_score)
print("Training ROC AUC Score:", train_roc_auc)

# Plotting the performance metrics
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
metrics_scores = [train_accuracy, train_precision, train_recall, train_f1_score, train_roc_auc]

plt.figure(figsize=(10, 6))
plt.bar(metrics_names, metrics_scores, color=['blue', 'green', 'orange', 'red', 'purple'])
plt.title('Model Performance on Training Data')
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.ylim(0, 1)  # Set y-axis limit to [0, 1] for better visualization
plt.show()