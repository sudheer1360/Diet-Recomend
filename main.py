from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pymongo import MongoClient

app = Flask(__name__)

cluster = MongoClient('mongodb://127.0.0.1:27017')
db = cluster['diet']
users = db['users']

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['post', 'get'])
def login():
    email = request.form['email']
    password = request.form['password']
    res = users.find_one({"email": email})
    if res and dict(res)['password'] == password:
        return render_template('index.html')
    else:
        return render_template('login.html', status='User does not exist or wrong password')

@app.route('/reg')
def reg():
    return render_template('signup.html')

@app.route('/regis', methods=['post', 'get'])
def register():
    name = request.form['name']
    email = request.form['email']
    password = request.form['password']
    k = {}
    k['name'] = name
    k['email'] = email
    k['password'] = password
    res = users.find_one({"email": email})
    if res:
        return render_template('signup.html', status="Email already exists")
    else:
        users.insert_one(k)
        return render_template('signup.html', status="Registration successful")

df = pd.read_csv("dataset.csv")

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Health Diseases'] = df['Health Diseases'].apply(lambda x: le.fit_transform([x])[0] if x else x)

X = df[['Age', 'Gender', 'Height(cm)', 'Weight(kg)', 'Sugar Level', 'Systolic_BP', 'Diastolic_BP', 'Health Diseases']]
y_diet = df[['Diet']]
y_exercise = df[['Exercise']]

X_train, X_test, y_train_diet, y_test_diet = train_test_split(X, y_diet, test_size=0.2, random_state=42)
X_train, X_test, y_train_exercise, y_test_exercise = train_test_split(X, y_exercise, test_size=0.2, random_state=42)

diet_model = RandomForestClassifier(n_estimators=100, random_state=42)
diet_model.fit(X_train, y_train_diet.values.ravel())

exercise_model = RandomForestClassifier(n_estimators=100, random_state=42)
exercise_model.fit(X_train, y_train_exercise.values.ravel())

def preprocess_input(input_data):
    le = LabelEncoder()
    input_data['Gender'] = le.fit_transform(input_data['Gender'])
    input_data['Health Diseases'] = le.fit_transform(input_data['Health Diseases'])

    return input_data

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        gender = int(request.form['gender'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        sugar_level = float(request.form['sugar_Level'])
        systolic_bp = float(request.form['systolic_BP'])
        diastolic_bp = float(request.form['diastolic_BP'])
        health_diseases = request.form['health_diseases']

        user_input = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Height(cm)': [height],
            'Weight(kg)': [weight],
            'Sugar Level': [sugar_level],
            'Systolic_BP': [systolic_bp],
            'Diastolic_BP': [diastolic_bp],
            'Health Diseases': [health_diseases]
        })

        processed_input = preprocess_input(user_input)
        diet_prediction = diet_model.predict(processed_input)[0]
        exercise_prediction = exercise_model.predict(processed_input)[0]
        return render_template('result.html', diet_prediction=diet_prediction, exercise_prediction=exercise_prediction)

if __name__ == '__main__':
    app.run(port=5001, debug=True)
