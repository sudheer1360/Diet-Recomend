import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle


df = pd.read_csv("dataset.csv")

print(df.columns)

le = LabelEncoder()

# Encode categorical columns (Gender and Health Diseases)
df['Gender'] = le.fit_transform(df['Gender'])
df['Health Diseases'] = df['Health Diseases'].apply(lambda x: ','.join(sorted(x.strip().split(', '))))
df['Health Diseases'] = le.fit_transform(df['Health Diseases'])

print(df["Health Diseases"])

# Split the data into features (X) and target (y)
X = df[['Age', 'Gender', 'Height(cm)', 'Weight(kg)', 'Health Diseases']]
y = df[['Diet Preferences']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model (you can use different metrics based on your use case)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy}')

# Save the trained model to a file (for later use)
with open('health_recommendation_model.pkl', 'wb') as model_file:
    pickle.dump(rf_model, model_file)

# Save the label encoders (for preprocessing during inference)
with open('label_encoders.pkl', 'wb') as le_file:
    pickle.dump({'Gender': le}, le_file)

# Now, you can use the saved model and label encoders for making predictions in another script or notebook.
