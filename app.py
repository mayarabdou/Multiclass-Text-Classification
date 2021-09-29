from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')




def clean_text(text):
    
    """
        text: a string
        
        return: modified strings in job titles
    """
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))
    
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub('', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    text = re.sub(r'\d', '', text)
    return text



@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    df = pd.read_csv('Job titles and industries.csv')
    df['job title'] = df['job title'].apply(clean_text)
    tf_vectorizer=TfidfVectorizer(ngram_range=(1,2))
    tf_vectorizer.fit(df['job title'])

    int_features =  request.form['text']
    
    final_features = clean_text(int_features)
    input=[final_features]
    data_title = tf_vectorizer.transform(input).toarray()
    prediction = model.predict(data_title)
    

    return render_template('index.html', prediction_text='Industry: {}'.format(prediction))



@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    text = request.args.get("text")
    prediction = predict(text)
    return jsonify(prediction=prediction)
    

if __name__ == "__main__":
    app.run(debug=True)