from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pymongo import MongoClient

cluster=MongoClient('mongodb://127.0.0.1:27017')
db=cluster['knee']
users=db['users']

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login',methods=['post','get'])
def login():
    user=request.form['username']
    password=request.form['password']
    res=users.find_one({"username":user})
    if res and dict(res)['password']==password:
        return render_template('index.html')
    else:
        return render_template('login.html',status='User does not exist or wrong password')


@app.route('/reg')
def reg():
    return render_template('signup.html')

@app.route('/regis',methods=['post','get'])
def register():
    username=request.form['username']
    password=request.form['password']
    k={}
    k['username']=username
    k['password']=password 
    res=users.find_one({"username":username})
    if res:
        return render_template('signup.html',status="Username already exists")
    else:
        users.insert_one(k)
        return render_template('signup.html',status="Registration successful")

# Load the dataset
df = pd.read_csv("dataset.csv")

# Encode categorical columns (Gender and Health Diseases)
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Health Diseases'] = df['Health Diseases']
df['Health Diseases'] = le.fit_transform(df['Health Diseases'])

# Split the data into features (X) and target (y)
X = df[['Age', 'Gender', 'Height(cm)', 'Weight(kg)', 'Health Diseases']]
y_health = df[['Diet Preferences']]
y_diet = df[['Exercise Recommendations']]

# Split the data into training and testing sets
X_train, X_test, y_train_health, y_test_health = train_test_split(X, y_health, test_size=0.2, random_state=42)
X_train, X_test, y_train_diet, y_test_diet = train_test_split(X, y_diet, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier for health model
diet_model = RandomForestClassifier(n_estimators=100, random_state=42)
diet_model.fit(X_train, y_train_health)

# Initialize and train the RandomForestClassifier for diet model
Exercise_model = RandomForestClassifier(n_estimators=100, random_state=42)
Exercise_model.fit(X_train, y_train_diet)

# Preprocess input data
def preprocess_input(input_data):
    input_data['Gender'] = input_data['Gender'].map({'Male': 0, 'Female': 1}).astype(int)
    input_data['Health Diseases'] = le.transform([input_data['Health Diseases']])[0]
    return input_data


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        gender = request.form['gender']
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        health_diseases = request.form['health_diseases']

        # Create a DataFrame with user input
        user_input = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Height(cm)': [height],
            'Weight(kg)': [weight],
            'Health Diseases': [health_diseases],
        })

        # Preprocess input data
        processed_input = preprocess_input(user_input)

        # Make predictions using the trained models
        Exercise_prediction = Exercise_model.predict(processed_input)[0]
        diet_prediction = diet_model.predict(processed_input)[0]

        return render_template('result.html', diet_prediction=diet_prediction, Exercise_prediction=Exercise_prediction)

if __name__ == '__main__':
    app.run(port=5001,debug=True)