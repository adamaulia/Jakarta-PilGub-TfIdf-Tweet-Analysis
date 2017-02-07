# -*- coding: utf-8 -*-
"""
Created on Sun Feb 05 13:13:34 2017

@author: Adam
"""
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def preprocessing_tweet(tweet):
    
    if isinstance(tweet, unicode):
        tweet = tweet.encode('utf-8', 'replace')
    if isinstance(tweet, (float, int, complex, long)):
        tweet = str(tweet)
        
#    tweet = tweet.encode("utf-8", "replace")
    #remove link/url (http)
    tw_remove_link=' '.join(re.sub(r"h\w+(:).\S+", " ", tweet.lower()).split())
    #print 'remove link',tw_remove_link
    
    #remove hashtag
    tw_remove_hashtag=' '.join(re.sub(r"(#).\S+", " ", tw_remove_link.lower()).split())
    #print 'remove hashtag',tw_remove_hashtag
    
    #remove mention @
    tw_remove_mention = ' '.join(re.sub(r"(@).\S+", " ", tw_remove_hashtag.lower()).split())
    #print 'remove mention',tw_remove_mention    
           
    #remove cc/via/
    tw_remove_ccvia = ' '.join(re.sub(r'\b(cc|RT|rt\b):?[ ](URL|@[^ ]+)', " ", tw_remove_mention.lower()).split())
    #tweet = re.sub(r'\b(via|cc|RT\b):?[ ](URL|@[^ ]+)', ' ', tweet)
    #print 'remove via',tw_remove_ccvia
    
    #remove punctuation
    tw_remove_puc = tw_remove_ccvia.strip(string.punctuation)
    #print 'remove punc',tw_remove_puc
    
    #remove stopword
    #tw_sw_removal = list(set(tw_remove_puc.split())-set(stopword))
    tw_sw_removal = filter(lambda x:x not in stopword, tw_remove_puc.split())
    
    return ' '.join(tw_sw_removal)
    #return tw_remove_puc
    

dataset=pd.read_excel('dataset.xlsx')
tweet=dataset['Tweet'].tolist()
date=dataset['Time'].tolist()

stopword= [line.rstrip('\n') for line in open('stopword.txt')]
print 'processing'
clean_tweet = []
for idx,item in enumerate(tweet):
    #print idx
    clean_tweet.append(preprocessing_tweet(item))

         
Tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=True, sublinear_tf=True)
Tfidf.fit(clean_tweet)

most_tweet=20
final_tweet = []
final_paslon = []
for tweet in dataset.groupby('Paslon'):
    prep_twt = tweet[1]['Tweet'].tolist()
    #print 'twt',prep_twt
    tmp = [preprocessing_tweet(item) for item in prep_twt]
        
    tw_by_paslon=np.sum(Tfidf.transform(tmp).toarray(),axis=0)
    sort_idx=np.argsort(tw_by_paslon.flatten())[::-1]
    most_paslon_tweet=[(Tfidf.get_feature_names()[item],tw_by_paslon[item]) for item in sort_idx.tolist()[:most_tweet]]

    final_tweet.append(dict(most_paslon_tweet))
    final_paslon.append(tweet[0])

# change 0 for AgusSylvi, 1 AhokDjarot, 2 AniesSandi
df = pd.DataFrame(final_tweet[1].items())
sns.plt.suptitle(final_paslon[1])
sns.barplot(x=0,y=1,data=df)     

#==============================================================================
# fig =  plt.figure()
# for idx,item in enumerate(final_tweet):
#     df = pd.DataFrame(item.items())
#     ax = fig.add_subplot(211)
#     sns.plt.suptitle(final_paslon[idx])
#     sns.barplot(x=0,y=1,data=df)
#==============================================================================
