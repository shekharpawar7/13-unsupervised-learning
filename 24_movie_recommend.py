# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 15:52:10 2023

@author: shekhar
"""

"""Q.The Entertainment Company, which is an online movie watching platform,
 wants to improve its collection of movies and showcase those that are 
 highly rated and recommend those movies to its customer by their movie
 watching footprint. For this, the company has collected the data and
 shared it with you to provide some analytical insights and also to
 come up with a recommendation algorithm so that it can automate its 
 process for effective recommendations. The ratings are between -9 and +9.

"""

'''
Business Objective:
    
maximize: Increase the visit number of customer on the online movie watching platfrom 
         by providing movies recommaendtion

Minimize: customer time for searching movies which like 

Constraints: base on the user rating DVD store can recommand the game 
and user can give good rating to game
'''

'''
Data Dictionary 
Id: it show the user id which the rating
Title : it show the movie name
Category : there some category of movies
Reviews : it show the reviews given by the customer
'''
import pandas as pd
movie=pd.read_csv("c:/4-DataSets/Entertainment.csv")

movie.shape
#there are 51 rows and 4 columns in the dataset

movie.isnull().sum()
"""
Id          0
Titles      0
Category    0
Reviews     0"""
#there is non  any null value

#check the datatype 
movie.dtypes
"""
Id            int64
Titles       object
Category     object
Reviews     float64
dtype: object"""
#the id column is int data type
#Titles column and Category has a object
#the  Category is Category data

movie.describe()
"""
                Id    Reviews
count    51.000000  51.000000
mean   6351.196078  36.289608
std    2619.679263  49.035042
min    1110.000000  -9.420000
25%    5295.500000  -4.295000
50%    6778.000000   5.920000
75%    8223.500000  99.000000
max    9979.000000  99.000000"""

movie.info()
"""
#   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   Id        51 non-null     int64  
 1   Titles    51 non-null     object 
 2   Category  51 non-null     object 
 3   Reviews   51 non-null     float64
dtypes: float64(1), int64(1), object(2)"""


from sklearn.feature_extraction.text import TfidfVectorizer
# It is going to craete TfidfVectorizer to sepaarte all stop words
tfidf=TfidfVectorizer(stop_words='english')


#check any null and fill with tha rating mean
movie['Category']=movie['Category'] .fillna('gernal')

# TF-IDF Vectorization
# now let us create  tfidf_matrix
tfidf_matrix=tfidf.fit_transform(movie.Category)
tfidf_matrix.shape #the shape is (51, 34)


from sklearn.metrics.pairwise import linear_kernel
#import liner_kernal to calaculate Cosine Similarity matrix
cosine_similarity_matrix=linear_kernel(tfidf_matrix,tfidf_matrix)
cosine_similarity_matrix.shape #the shape is (51, 51)


# Mapping movies Names to Indixs
movie_index = pd.Series(movie.index, index=movie['Titles']).drop_duplicates()


#set some example for input
movie_name='Four Rooms (1995)'
topN=10


# Get the index of the given movie name
movie_id = movie_index[movie_name]
movie_id #spider-man movie game id is 17
        
# Calculate cosine similarity scores with all remmaning data of Four Rooms (1995) movie
cosine_scores = list(enumerate(cosine_similarity_matrix[movie_id]))
        
# Sort the scores in descending order which high score is on top
sorted_cosine_scores = sorted(cosine_scores)
        
# Get the top N most similar games
cosine_scores_N = cosine_scores[0:topN+1]
        
# Extract indexs and scores usin for loop
movie_idx = [i[0] for i in cosine_scores_N]
movie_scores = [i[1] for i in cosine_scores_N]
        
# Create a DataFrame to display recommendations

#let create one empty datafream with Titles and score colume 
movie_similar_show = pd.DataFrame(columns=['Titles', 'score'])

#let store the movie name in empty datafream by using movie idx
movie_similar_show['Titles'] = movie.loc[movie_idx, 'Titles']

#store the score created by cosine similarity matrix
movie_similar_show['score'] = movie_scores
movie_similar_show.reset_index(inplace=True)

#print Top recommended movies        
print(movie_similar_show)




