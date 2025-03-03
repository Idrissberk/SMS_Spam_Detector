import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import zipfile
import requests
from io import BytesIO

# Dataset URL
zip_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'

# Downloading the zip file and extracting the CSV file
response = requests.get(zip_url)
with zipfile.ZipFile(BytesIO(response.content)) as z:
    # Assuming the CSV file you want to read is named 'SMS Spam Collection'
    csv_file = z.open('SMSSpamCollection')
    sms_data = pd.read_csv(csv_file, sep='\t', names=['label','message'])

# Convert the labels to binary (0 for ham, 1 for spam)
sms_data['label'] = sms_data['label'].map({'ham':0, 'spam':1})

#Splitting the data into training and testing sets
x = sms_data['message']
y = sms_data['label']

#Vectorizing the test data
vectorized = TfidfVectorizer()
x_vectorized = vectorized.fit_transform(x)

#Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_vectorized, y, test_size=0.2, random_state=42)

#Creating and Training the model

model = SVC(kernel='linear')
model.fit(x_train, y_train)

#Making predictions on the test data
y_pred = model.predict(x_test)

#Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Accuracy: ", accuracy)
print("Classification Report: \n", report)