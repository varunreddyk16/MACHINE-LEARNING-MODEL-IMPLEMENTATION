# MACHINE-LEARNING-MODEL-IMPLEMENTATION #

COMPANY : CODTECH IT SOLUTIONS

NAME : KUNCHARAPU VARUN REDDY

INTERN ID : CTO8SGE

DOMAIN : PYTHON PROGRAMMING

DURATION :4 WEEKS

MENTOR : NEELA SANTHOSH

DESCRIPTION OF THE PROJECT

The code is a program that checks if an email is spam or not. It uses a dataset of emails, where each email is labeled as "spam" (unwanted) or "ham" (normal). The program learns from this dataset to predict whether new emails are spam or not.

Data Loading and Cleaning:

The dataset (mail_data.csv) is loaded using pandas. It contains two columns: Category (spam or ham) and Message (the email content).

Missing values are handled by replacing them with empty strings.

Label Encoding

The Category column is converted into numerical values: spam is mapped to 0, and ham is mapped to 1. This makes it easier for the machine learning model to process the data.

Data Splitting

The dataset is split into training and testing sets using train_test_split from scikit-learn. This ensures that the model is trained on one portion of the data and evaluated on another.

Text to Numerical Conversion:

The email messages  are converted into numerical features using TfidfVectorizer. This technique calculates the importance of each word in the email relative to the entire dataset, creating a matrix of TF-IDF (Term Frequency-Inverse Document Frequency) values.

Model Training:

A Logistic Regression model (from scikit-learn) is trained on the processed training data. Logistic Regression is a simple yet effective algorithm for binary classification tasks like spam detection.

Model Evaluation:

The model’s performance is evaluated using the accuracy_score metric. It calculates the percentage of correctly classified emails on both the training and test datasets.

Prediction:

The trained model can predict whether a new email is spam or ham. For example, if you input an email like "Congratulations! You've won a prize!", the model will classify it as either spam (0) or ham (1).

Modules 
pandas: For loading and cleaning the dataset.

numpy: For numerical operations (though not heavily used in this code).

scikit-learn: For machine learning tasks, including:

train_test_split: Splitting the dataset into training and testing sets.

TfidfVectorizer: Converting text data into numerical features.

LogisticRegression: Training the spam detection model.

accuracy_score: Evaluating the model’s performance.

As i,m new to machine learning i have taken the help of online sources to complete this project and i used collab to build this.It feel productive to learn new things .

# EMAIL_DATASET IN CSV FORMAT #
[mail_data.csv](https://github.com/user-attachments/files/19188621/mail_data.csv)

# OUTPUT #
![Image](https://github.com/user-attachments/assets/ce252d02-da2c-4eef-8382-d3d61321eee3)

# COLLAB NOTEBOOK COPY IS AVAILABLE IN CODE
# THANK YOU FOR READING #


