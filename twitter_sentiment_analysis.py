

#data manipulation library
import pandas as pd
import nltk

#reading the dataset
tweet_df = pd.read_csv('/content/drive/MyDrive/projects/Tweets.csv')
tweet_df.head()

### Data Sanity checks
shape = tweet_df.shape
print('The tweet dataset has', shape[0], 'rows')
print('The tweet dataset has', shape[1], 'columns')

#checking missing values
tweet_df.isnull().sum()

#checking the datatypes of the columns
tweet_df.dtypes

#checking for duplicates
tweet_df.duplicated().sum()

#checking the dataset info
tweet_df.info()

#checking the statistical summary of the numeric column'
tweet_df.describe().T

#checking the statistical summary of the text columns
tweet_df.describe(include = 'object').T

### EDA
#Importing data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

#showing the flight distribution
plt.figure(figsize = (7,4))
sns.countplot(data = tweet_df, x = 'airline')
plt.title('Airline distribution')
plt.xticks(rotation = 45)
plt.show()

#visualizing the sentiments
sentiments = tweet_df.airline_sentiment.value_counts()
plt.pie(sentiments, labels = sentiments.index, autopct='%1.1f%%', startangle=90)
plt.title('Airline Sentiments Distribution')
plt.show()

#plotting the negative reason sentiment to see which reason was leading
neg_sentiments = tweet_df.negativereason.value_counts()
plt.pie(neg_sentiments, labels = neg_sentiments.index)
plt.title('Reasons for the negative sentiments')
plt.show()

### Feature Engineering
#dropping duplicates
tweet_df.drop_duplicates(inplace = True)
print(tweet_df.duplicated().sum())

tweet_df.head()

#creating a dataframe of the columns needed
df = tweet_df[['airline_sentiment', 'text']]
df.head()

#importing a label encoder
from sklearn.preprocessing import LabelEncoder
#initializing the encoder
le = LabelEncoder()
#encoding the data
df['target'] = le.fit_transform(df['airline_sentiment'])
df.head()

X = df.drop(['target', 'airline_sentiment'], axis = 1)
y = df.pop('target')

### Model Building
#importing the necessary
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

#splitting the data to training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1, stratify = y)
X_train.head()

vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
X_train_tfidf = vectorizer.fit_transform(X_train['text'])
X_test_tfidf = vectorizer.transform(X_test['text'])

print("X_train_tfidf shape:", X_train_tfidf.shape)
print("y_train shape:", y_train.shape)

### Fitting a logistic regression model
#initializing the model
logreg = LogisticRegression(multi_class = 'ovr', max_iter = 500)
#fitting the data on the training set
logreg.fit(X_train_tfidf, y_train)

preds = logreg.predict(X_test_tfidf)
#getting the accuracy score
logreg_accuracy = accuracy_score(y_test, preds)
print("The accuracy of the model is: ", logreg_accuracy)

#printing the classification report
report = classification_report(y_test, preds)
print(report)

### Using SGD model to fit the data and make predictions on it
#importing the model
from sklearn.linear_model import SGDClassifier

#initializing the model
sgd = SGDClassifier(loss='log', random_state=42, max_iter=1000, tol=1e-3)

# Fitting the model on the training data
sgd.fit(X_train_tfidf, y_train)

#making predictions on the test data
sgd_pred = sgd.predict(X_test_tfidf)

#getting the accuracy score
sgd_accuracy = accuracy_score(y_test, sgd_pred)
print(sgd_accuracy)

#pritning the classification report
sgd_report = classification_report(y_test, sgd_pred)
print(sgd_report)
