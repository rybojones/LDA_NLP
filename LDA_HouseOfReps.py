#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.parsing.preprocessing import remove_stopwords
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora


# In[2]:


# read in house of reps tweets from csv file
congress_csv = pd.read_csv('congress.csv')


# In[3]:


# visualize data before performing any actions
print(congress_csv['Tweet'].head())


# In[4]:


print(congress_csv['Tweet'][1])


# In[5]:


# make sure strings are in unicode format by forcing type
congress_csv['Tweet'] = congress_csv['Tweet'].astype('unicode')


# In[6]:


# perform first step of cleaning
# remove all stop words from tweets
cleaned_tweets = [remove_stopwords(token) for token in congress_csv['Tweet']]


# In[7]:


# verify stop words were removed by comparing
# to a tweet that was printed previously
print(cleaned_tweets[0])


# In[8]:


# initialize empty list to store all tweets with
# stop words removed
tweet_list = []


# In[9]:


# build tokenizer and tokenzie all tweets
# also store into. new variable
# also split each tweet into it's
# individual words so more cleaning can be done
tokenizer = RegexpTokenizer(r'\w+')
for row in range(len(cleaned_tweets)):
    tweet_list.append(cleaned_tweets[row]) 
    tweet_list[row] = tokenizer.tokenize(cleaned_tweets[row])


# In[10]:


# remove words that are only one char
tweet_list = [[word for word in row if len(word) > 1] for row in tweet_list]

# convert all text to lowercase
tweet_list = [ [ word.lower() for word in tweet ] for tweet in tweet_list ]

# remove link text from tweets by removing
# any words that contain "http"
tweet_list = [ [ word for word in row if not ('http' in word) ] for row in tweet_list ]          
       
# remove & word (amp) that was prevously
# displaying in results of topic model     
tweet_list = [ [ word for word in row if word != 'amp'] for row in tweet_list ]


# In[11]:


# verify all above cleaning
# steps were accurately performed
print(tweet_list[5])


# In[12]:


# lemmatize words
lemmatizer = WordNetLemmatizer()
lemma_tweets = [[lemmatizer.lemmatize(token) for token in row] for row in tweet_list]


# In[13]:


# verify lemmatizer worked
for x in range(0,3):
    print(lemma_tweets[x])


# In[14]:


# verify lemmatizer worked
bigram = Phrases(lemma_tweets, min_count=15)
for tweet in range(len(lemma_tweets)):
    for token in bigram[lemma_tweets[tweet]]:
        if '_' in token:
            lemma_tweets[tweet].append(token)


# In[15]:


# build dictionary of lemmatized words in dataset
dictionary = Dictionary(lemma_tweets)

# filter out words that occur less than 20 documents,
# or more than 75% of the documents.
dictionary.filter_extremes(no_below=20, no_above=0.75)


# In[16]:


# build corpus to pass into LDA model
corpus_congress = [dictionary.doc2bow(tweet) for tweet in lemma_tweets]


# In[17]:


# view number of tokens in dictionary
# and tweets in corpus
print('Number of unique tokens: %d' % len(dictionary))
print('Number of tweets: %d' % len(corpus_congress))


# In[18]:


# set training parameters
num_topics = 5
chunksize = 2000
passes = 20
iterations = 400
eval_every = None  

temp = dictionary[0]
id2word = dictionary.id2token

# build model
LDA_model = LdaModel(
    corpus=corpus_congress,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)


# In[19]:


# find top topics from model
top_topics = LDA_model.top_topics(corpus = corpus_congress, topn=5)

# find average coherence of topics
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)

# print top topics
from pprint import pprint
pprint(top_topics)

