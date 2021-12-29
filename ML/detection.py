import pandas as pd
import numpy as np
#import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pickle
#nltk.download('stopwords')
def getData():
    trueData = pd.read_csv('True.csv')
    fakeData = pd.read_csv('Fake.csv')
    
    fakeData['label'] = [0] * len(fakeData['text'])
    trueData['label'] = [1] * len(trueData['text'])
    all_data = pd.concat([fakeData, trueData])
    #print(all_data.columns)
    all_data.drop(columns=['subject','date'], inplace=True)
    all_data = all_data.sample(frac=1)
    #print(all_data)
    return all_data

def preprocessData(data):
    print("PREPROCESSING")
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    data['text'] = data['text'].apply(lambda x: x.lower())
    data['text'] = data['text'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
    
    data['text'] = data['text'].apply(lambda x: [ps.stem(word) for word in x.split() if not word in stop_words])
    data['text'] = data['text'].apply(lambda x: ' '.join(x))
    return data

def FeatureExtraction(data):
    print('FEATURE EXTRACTION')
    x_train, x_test, y_train, y_test=train_test_split(data['text'], data['label'], test_size=.2)
    tfidf_vectorizer = TfidfVectorizer(max_df=.7)
    tfidf_train = tfidf_vectorizer.fit_transform(x_train) # learns mean and variance of our training set FEATURES
    tfidf_test = tfidf_vectorizer.transform(x_test) # uses said mean and variance on our testing set FEATURES
    
    return tfidf_vectorizer, tfidf_train, tfidf_test, y_train, y_test

def pac_predict(tfidf_train, tfidf_test, y_train, y_test):
    # train model
    pac = PassiveAggressiveClassifier(max_iter=100)
    print('TRAINING:')
    pac.fit(tfidf_train, y_train)

    print('PREDICTING')
    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {100 * score}%') # 99% accuracy
    return pac

def MNB_predict(tfidf_train, tfidf_test, y_train, y_test):
    MNB = MultinomialNB()
    print("TRAINING")
    MNB.fit(tfidf_train, y_train)

    print("PREDICTING")
    y_pred = MNB.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {100 * score}%') 
    return MNB

def main():
    data = preprocessData(getData())
    tfidf_vectorizer, tfidf_train, tfidf_test, y_train, y_test = FeatureExtraction(data)
    pacModel = pac_predict(tfidf_train, tfidf_test, y_train, y_test)
    mnbModel = MNB_predict(tfidf_train, tfidf_test, y_train, y_test)
    
    with open('vectorizer.pk', 'wb') as fp:
        pickle.dump(tfidf_vectorizer, fp)
    with open('pac_model.pk', 'wb') as fp:
        pickle.dump(pacModel, fp)
    with open('mnb_model.pkl', 'wb') as fp:
        pickle.dump(mnbModel, fp)




if __name__ == '__main__':
    main()

