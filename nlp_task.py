#!/usr/bin/env python
# coding: utf-8

# ## **Import libraries**

# In[1]:


import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import re
get_ipython().run_line_magic('matplotlib', 'inline')
import pickle


# In[2]:


import nltk
from nltk import word_tokenize
from nltk import ngrams
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud 


# ## **Read and Explore data**

# In[3]:


df = pd.read_csv('Job titles and industries.csv')
df = df[pd.notnull(df['industry'])]


# In[4]:


df.head(10)


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


df.columns


# In[9]:


print(df['job title'].apply(lambda x: len(x.split(' '))).sum())


# ## **Functions**

# In[10]:


def print_element(index):
    """
    Have a look on few job titles and industry pairs
    
    """
    example = df[df.index == index][['job title', 'industry']].values[0]
    if len(example) > 0:
        print(example[0])
        print('industry:', example[1])


# In[11]:


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        
        return: modified strings in job titles
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub('', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    text = re.sub(r'\d', '', text)
    return text


# In[12]:


stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    """
    Word cloud for job titles 
    
    in which the size of each word indicates 
    
    its frequency or importance. 
    
    """
    wordcloud = WordCloud(
        background_color='black',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
).generate(str(data))

    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()


# In[13]:


def top_tfidf_feats(row, features, top_n=20):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

    """
    
    Frequently occurring terms[job titles] 
    
    for each class[Industry]
    
    """

def top_feats_in_doc(Xtr, features, row_id, top_n=20):
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)

def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=10):
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=20):
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs

def plot_tfidf_classfeats_h(dfs, num_class=4):
    
    fig = plt.figure(figsize=(12, 100), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        #z = int(str(int(i/3)+1) + str((i%3)+1))
        ax = fig.add_subplot(num_class, 1, i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=16)
        ax.set_ylabel("Word", labelpad=16, fontsize=16)
        ax.set_title("Class = " + str(df.label), fontsize=25)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.feature)
        
        for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(20) 
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()


# ## **Data Visualization**

# #### Word cloud to show the importance of words according to its size:

# In[14]:


show_wordcloud(df['job title'])


# In[15]:


vectorizer=TfidfVectorizer(ngram_range=(1,2))


# In[16]:


job_title_vectorized = vectorizer.fit_transform(df['job title'])


# #### **Frequently occured job title in each class:**

# In[17]:


class_y = df['industry']
class_features = vectorizer.get_feature_names()
class_top_dfs = top_feats_by_class(job_title_vectorized, class_y, class_features)
plot_tfidf_classfeats_h(class_top_dfs, 7)


# In[ ]:





# ## **Data Preprocessing and Cleaning**

# In[18]:


print(df['job title'].apply(lambda x: len(x.split(' '))).sum())


# In[19]:


print_element(250)


# In[20]:


df['job title'] = df['job title'].apply(clean_text)
print_element(250)


# In[21]:


clean_text('salesforce consultant / business analyst')


# In[22]:


print(df['job title'].apply(lambda x: len(x.split(' '))).sum())


# In[23]:


tf_vectorizer=TfidfVectorizer(ngram_range=(1,2))


# **vectorizing data:**

# In[24]:


data = tf_vectorizer.fit_transform(df['job title'])


# In[25]:


data_ = pd.DataFrame(data.toarray(), columns=np.array(tf_vectorizer.get_feature_names()))


# In[26]:


data_


# ## **Split Data**

# In[27]:


X = data_
y = df.industry
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)


# In[28]:


type(X_train)


# ## **Model Train**

# #### **Multinomial Naive Baise**

# In[29]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
my_industries = ['IT', 'Marketing', 'Education', 'Accountancy']

# nb = Pipeline([('vect', CountVectorizer()),
#                ('tfidf', TfidfTransformer()),
#                ('clf', MultinomialNB()),
#               ])
nb = MultinomialNB()
nb.fit(X_train, y_train)

from sklearn.metrics import classification_report
y_pred = nb.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=my_industries))


# #### **RandomForestClassifier**

# In[34]:


from sklearn.ensemble import RandomForestClassifier

tfidf = TfidfVectorizer(stop_words='english')


rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(rf.score(X_test, y_test))
predictions = rf.predict(X_test)
# print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


# #### **Support Vector Machine** 

# In[35]:


from sklearn import svm
from sklearn.metrics import classification_report

clf = svm.LinearSVC()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(clf.score(X_test,y_test))
print(classification_report(y_test, y_pred))


# **Observation: SVM is the Best Model**

# In[36]:


X_train.shape


# In[37]:


# Saving model to disk
pickle.dump(clf, open('model.pkl','wb'))


# In[38]:


# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))


# In[39]:


result = model.score(X_test, y_test)
print(result)


# #### **Testing the model on some job titles:**

# In[40]:


def classify(jobTitle_text):
    global model
    global tf_vectorizer
    global data
    jobTitle=tf_vectorizer.transform([jobTitle_text])
    labels=df['industry']
    categories=df['industry']
    look_up=dict(zip(labels,categories))
    return look_up[clf.predict(jobTitle)[0]]


# In[41]:


artice_text="teacher"


# In[42]:


classify(artice_text)

