# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 15:52:10 2023

@author: shekhar
"""
"""
Q) Build a recommender system with the given data using UBCF.

This dataset is related to the video gaming industry and a survey was 
conducted to build a 
recommendation engine so that the store can improve the sales of its 
gaming DVDs. Snapshot of the dataset is given below. Build a 
Recommendation Engine and suggest top selling DVDs to the store customers.
"""

'''
Business Objective:
    
maximize: Identify the game and user id relation and recommand
 the game base user rating 

Minimize: in this minimize the DVDs store stock base on user rating store person can
sale the DVDs of game. 

Constraints: base on the user rating DVD store can recommand the game 
and user can give good rating to game
'''


'''
Data Dictionary 
userId : this is user id of customer 
game: it show the name of game
rating: it show the rating of game it given by the user

'''

import pandas as pd
game=pd.read_csv('C:/4-DataSets/game.csv')

game.columns
#there is only there column in dataset 'userId', 'game', 'rating'

game.shape
#dataset having a 5000 rows and 3 column

game.info()
'''
0   userId  5000 non-null   int64  
1   game    5000 non-null   object 
2   rating  5000 non-null   float64

there no any null value in any col
'''

game.dtypes
"""
userId is  int64 datatype
game is object datatype
rating is float64 datatype
"""

from sklearn.feature_extraction.text import TfidfVectorizer
# It is going to craete TfidfVectorizer to sepaarte all stop words
tfidf=TfidfVectorizer(stop_words='english')

#check any null and fill with tha rating mean
game['rating']=game['rating'] .fillna(game.rating.mean()) 

# TF-IDF Vectorization
# now let us create  tfidf_matrix
tfidf_matrix=tfidf.fit_transform(game.game)
tfidf_matrix.shape #the shape is 5000 X 3068

from sklearn.metrics.pairwise import linear_kernel
#import liner_kernal to calaculate Cosine Similarity matrix
cosine_similarity_matrix=linear_kernel(tfidf_matrix,tfidf_matrix)
cosine_similarity_matrix.shape #the shape is 5000X5000

# Mapping Game Names to Indices
game_index = pd.Series(game.index, index=game['game']).drop_duplicates()


#set some example for input
game_name='Spider-Man: Web of Shadows'
topN=10

# Get the index of the given game name
game_id = game_index[game_name]
game_id #spider-man movie game id is 4841
        
# Calculate cosine similarity scores with all remmaning data of spider movie
cosine_scores = list(enumerate(cosine_similarity_matrix[game_id]))
        
# Sort the scores in descending order which high score is on top
sorted_cosine_scores = sorted(cosine_scores)
        
# Get the top N most similar games
cosine_scores_N = cosine_scores[0:topN+1]
        
# Extract indexs and scores usin for loop
game_idx = [i[0] for i in cosine_scores_N]
game_scores = [i[1] for i in cosine_scores_N]
        
# Create a DataFrame to display recommendations
game_similar_show = pd.DataFrame(columns=['game', 'score'])
game_similar_show['game'] = game.loc[game_idx, 'game']
game_similar_show['score'] = game_scores
game_similar_show.reset_index(inplace=True)

#print Top recommended movies        
print(game_similar_show)




