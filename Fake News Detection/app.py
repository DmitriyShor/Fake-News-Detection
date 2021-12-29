from flask import Flask, render_template, request
import requests
import pickle
import re
from newspaper import Article
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

app = Flask(__name__)

@app.route("/", methods = ['GET', 'POST'])
def getData():
    #prediction = -1
    if request.method == 'POST':
        url = request.form.get('url')
        print(f'URL: {url}')
        # return url
        text, status = webscrape(url)
        if status == -1:
            return render_template('form.html', value='Check website URL')
        prediction = predict(text)
        print(prediction)
        output = ""
        if prediction:
            output = 'The site: ' + url + ' is a credible news article.'
        else:
            output = 'The site: ' + url + ' is an uncredible news article.'
        return render_template('form.html', value=output)
        
    return render_template("form.html")

# def update(data):
#     print('Current Value', data['value'])

def webscrape(url):
    
        #page = requests.get(url)
    print(url)
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = preprocess(article.text)
        return text, 0
    except:
        print('ERROR FOR LINK:', url)
        return 'none', -1
    #predict(text)

def preprocess(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = text.lower() # lower case
    text = text.strip() # remove new lines
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = [ps.stem(word) for word in text.split() if not word in stop_words]
    text = ' '.join(text)
    return text

def predict(text):
    with open(r'C:\\Users\\dima\\OneDrive\\Desktop\\Fake-News-Detection\\ML\\vectorizer.pk', 'rb') as vectorizerFile:
        vectorizer = pickle.load(vectorizerFile)
    with open(r'C:\\Users\\dima\\OneDrive\\Desktop\\Fake-News-Detection\\ML\\pac_model.pk', 'rb') as pacFile:
        pac_model = pickle.load(pacFile)
    with open(r'C:\\Users\\dima\\OneDrive\\Desktop\\Fake-News-Detection\\ML\\mnb_model.pk', 'rb') as mnbFile:
        mnb_model = pickle.load(mnbFile)
    vectorizedInput = vectorizer.transform([text])
     
    pac_pred = pac_model.predict(vectorizedInput) # overfit on our data
    mnb_pred = mnb_model.predict(vectorizedInput)
    
    #print(pac_pred)  
    #print(mnb_pred[0])
    return mnb_pred[0] # only using mnb as it predicts correctly
    
if __name__ == "__main__":
    app.run()