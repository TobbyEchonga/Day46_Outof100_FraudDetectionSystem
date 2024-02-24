import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Sample dataset (replace with your own dataset)
data = {
    'Amount': [100, 150, 200, 50, 120, 180, 250, 30, 40, 300, 25, 160],
    'IsFraud': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
}

df = pd.DataFrame(data)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['Amount']], df['IsFraud'], test_size=0.2, random_state=42)

# Train an Isolation Forest model
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Visualize the results
plt.scatter(X_test, y_test, c=predictions, cmap='viridis')
plt.xlabel('Amount')
plt.ylabel('IsFraud')
plt.title('Fraud Detection Results')
plt.show()

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%\n')

print('Classification Report:')
print(classification_report(y_test, predictions))
