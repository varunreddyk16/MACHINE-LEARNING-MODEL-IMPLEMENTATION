import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv(r'/content/mail_data.csv')

# Replace null values with an empty string
data = df.where((pd.notnull(df)), '')

# Convert 'spam' to 0 and 'ham' to 1
data.loc[data['Category'] == 'spam', 'Category'] = 0
data.loc[data['Category'] == 'ham', 'Category'] = 1

x = data['Message']
y = data['Category']

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=3)


feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

# Transform the training and testing data
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Convert the labels to integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Predict on the training data
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print("Accuracy on training data:", accuracy_on_training_data)

# Predict on the test data
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print("Accuracy on test data:", accuracy_on_test_data)

# Predict on a sample email
input_your_mail = ["Congratulations! You've won a $1000 Walmart gift card. Click here to claim your prize."]
input_data_features = feature_extraction.transform(input_your_mail)
prediction = model.predict(input_data_features)

if prediction[0] == 1:
    print("The mail is classified as: Ham Mail")
else:
    print("The mail is classified as: Spam Mail")
