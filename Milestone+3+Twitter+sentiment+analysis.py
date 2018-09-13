
# coding: utf-8

# In[120]:


import twitter
import tweepy
import csv
import re


# In[121]:


ckey = "xDEONT6heklsgaLxfgq9P1Mel"
csecret = "1IDXFdRc3CRzIOMJJJ2nBnVOOTc6b8L8rJ8YX82BvKiyshwB2X"
atoken = "990328439323447296-IpoCBsOcZZzHGZisVUZz0AI1QN1H8qB"
asecret = "Z1KE8rzdXSeTqEGQ7kyQsMiLt7swaNfB279Je78Np5qgx"
OAUTH_KEYS = {'consumer_key':ckey,'consumer_secret':csecret,'access_token_key':atoken,'access_token_secret':asecret}
auth=tweepy.OAuthHandler(OAUTH_KEYS['consumer_key'],OAUTH_KEYS['consumer_secret'])
api = tweepy.API(auth)
testTweet = tweepy.Cursor(api.search,q='lonely OR depressed OR unhappy OR will to live OR suicide OR kill myself',lang="en").items(2000)
csvFile = open('result.csv', 'w')
csvWriter = csv.writer(csvFile)
csvWriter.writerow(["date of tweet","tweet content","username","distress_level"])


# In[30]:


dir(subTweet)


# In[122]:


for tweet in testTweet:
    text = re.sub(r"http\S+","", tweet.text.encode('ascii','ignore').decode('utf=8'))
    date = tweet.created_at
    username = tweet.user.screen_name
    csvWriter.writerow([date, text, username])


# In[123]:


csvFile.close()


# In[124]:


import pandas as pd


# In[125]:


data = pd.read_csv("result.csv")


# In[140]:


def classify(x):
    subTweet = tweepy.Cursor(api.user_timeline,screen_name = x).items(200)
    level=0
    for status in subTweet:
        #print(status.text)
        if (("sad" in status.text) or ("kill" in status.text) or ("depressed" in status.text) or ("lonely" in status.text)):
        #if (("kill" in status.text):
            print(status.text)
            level=level+1
            print(level)
    if (level>3):
        return 2
    elif (level==1 or level==2):
        return 1
    else:
        return 0   


# In[117]:


classify('IssOkayHun')


# In[134]:


len(data)


# In[141]:


csvFile = open('training.csv', 'w')
csvWriter = csv.writer(csvFile)
csvWriter.writerow(["date of tweet","tweet content","username","distress_level"])


# In[142]:


for iterator in range(0,len(data)-1):
    x = data.iloc[iterator,2]
    y = classify(x)
    csvWriter.writerow([data.iloc[iterator,0],data.iloc[iterator,1],data.iloc[iterator,2], y])


# In[145]:


csvFile.close()


# In[146]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords


# In[147]:


trainData = pd.read_csv("training.csv")


# In[148]:


trainData


# In[150]:


trainData['text length'] = trainData['tweet content'].apply(len)
trainData.head()


# In[151]:


g = sns.FacetGrid(data=trainData, col='distress_level')
g.map(plt.hist, 'text length', bins=50)


# In[152]:


sns.boxplot(x='distress_level', y='text length', data=trainData)


# In[153]:


distress = trainData.groupby('distress_level').mean()
distress.corr()


# In[154]:


train_class = trainData[(trainData['distress_level'] == 1) | (trainData['distress_level'] == 0)]
train_class.shape


# In[155]:


X = train_class['tweet content']
y = train_class['distress_level']


# In[156]:


import string
def text_process(text):
    '''
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Return the cleaned text as a list of words
    '''
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[157]:


sample_text = "Hey there! This is a sample tweet, which happens to contain punctuations."
print(text_process(sample_text))


# In[159]:


import sklearn


# In[161]:


from sklearn.feature_extraction.text import CountVectorizer


# In[162]:


bow_transformer = CountVectorizer(analyzer=text_process).fit(X)


# In[163]:


len(bow_transformer.vocabulary_)


# In[164]:


level_25 = X[24]


# In[165]:


level_25


# In[166]:


bow_25 = bow_transformer.transform([level_25])


# In[167]:


bow_25


# In[168]:


print(bow_25)


# In[169]:


X = bow_transformer.transform(X)


# In[170]:


print(X)


# In[171]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[172]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)


# In[173]:


preds = nb.predict(X_test)


# In[174]:


from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, preds))
print('\n')
print(classification_report(y_test, preds))

