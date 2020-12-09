# MusicRecommendation

This music recommendation task concentrates on two different ways for recommendation:
1. **Popularity-based**, which is used for new user who didn't have music listened record 
2. **Item-based collabrative filtering**, which is used for regular user who has music listened record. This approach uses Jaccard similarity coefficient  

## Data Exploration
This instruction focuses on how to make recommendation, for the topic that how to explore data please refer to the code file.
The structure of dataframe which is used in the recommendation shows as below:

## Popularity-base Recommendation
Using a top list for recommendation for new user, because those users don't have listened record so that can't apply collabrative filtering on recommendation for them.
``` python
# recommend by popularity for new user who doesn't have listening record
def popularity_recommendation(data, user_id, item_id,recommend_num):
    # based on the item_id, get the popular items
    popular = data.groupby(item_id)[user_id].count().reset_index()
    # rename groupby column
    popular.rename(columns = {user_id:"score"},inplace = True)
    # sort the data
    popular = popular.sort_values('score',ascending=False)
    # create rank
    popular['rank'] = popular['score'].rank(ascending=0, method='first')
    return popular.set_index('rank').head(recommend_num)
```
user_id : the column name of user, it's 'user'  
item_id : can be the column name of the song, artist, or a release, it depends on what you want to recommend.  
recommend_num : how many items you want to recommend to the user, for example like 10 or 100  

First, find out the the popularity based on the item_id, so that we have the popular ordered list. And then we select the top recommend_num items from the list and recommend to the user.  

## Item-based collabrative filtering
For a regular user, we can have data of what song he has listened. Based on Jaccard similarity coefficient, we can calculate the average similarity between a specific song and all songs the user listened.

For example, we have song A and song B, the Jaccard similarity coefficient is calculated like this:
1. get the the amount of users who listened both song A and Song X, which is A∩X
2. get the union of amount of users who listened to song A and amount of users who listened to song X , which is A∪X
3. Jaccard similarity coefficient between A and B is equal to : (A∩X)/(A∪X)

If the user whom we want to recommend to has listened 3 songs: A,B,C. And we want to know should we recommend song X to the user. We need to calculate the Jaccard similarity coefficient between each song the user listened with song X. so we get 3 Jaccard similarity coefficient， J(A), J(B), J(C), and we get the average of them J(X) = (J(A) + J(B) +J(C))/3.

When there is another song Y we consider whether recommend, we calculate J(Y) as above. Then we compare J(X) and J(Y)， whose value is larger, who is more highly reommended.  

```python
def construct_cooccurence_matrix(self, user_items, all_items):
        
        # get all the users which listened to the songs that the certain user listened to
        user_item_users = []
        for i in user_items:
            users = self.get_item_users(i)
            user_item_users.append(users)
        self.cooccurence_matrix = np.zeros((len(user_items),len(all_items)),float)
        print(self.cooccurence_matrix.shape)
        
        # calculate the similarity between user listened songs and all songs in the training data
        # using Jaccard similarity coefficient
        for i in range(0, len(user_items)):
            
            # get users of a certain listened song of a certain user
            user_listened_certain = set(user_item_users[i])
            
            for j in range(0, len(all_items)):
                user_unique = self.get_item_users(all_items[j])
                user_intersection = user_listened_certain.intersection(user_unique)
                if len(user_intersection)!=0:
                    user_union = user_listened_certain.union(user_unique)
                    self.cooccurence_matrix[i][j] = float(len(user_intersection)/len(user_union))
                else:
                    self.cooccurence_matrix[i][j] = 0
        return self.cooccurence_matrix
  ```
From construct_cooccurence_matrix function, we construct the matrix for calculation of Jaccard coefficient between songs. like if the user whom we want to make recommendation to has listened 3 songs, and 5 other songs we consider whether recommend. Then this matrix will be 3*5, each row means a song the user listened, each column means a song we consider whether recommend.

```python
# use cooccurence matrix to make top recommendation
    def generate_top_recommendation(self, user, all_songs, user_songs):
        print("Non Zero values in cooccurence %d" % np.count_nonzero(self.cooccurence_matrix))
        # get average similarity between all the listened songs and a certain song
        scores = self.cooccurence_matrix.sum(axis=0)/float(self.cooccurence_matrix.shape[0])
        print("score's shape: {n}".format(n=scores.shape))
        scores = scores.tolist()
        
        sort_index = sorted(((e,i) for (i,e) in enumerate(scores)),reverse=True)
        
        col = ['user_id', 'song', 'score', 'rank']
        df = pd.DataFrame(columns=col)
        rank = 1 
        for i in range(0,len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank <= 10:
                df.loc[len(df)]=[user,all_songs[sort_index[i][1]],sort_index[i][0],rank]
                rank = rank+1
        
        if df.shape[0] == 0:
            print("The current user has no songs for training the item similarity based recommendation model.")
            return -1
        else:
            return df
  ```
As we already have the Jaccard coefficient matrix, we can calculate the average coefficient value between the song we consider reommend and the songs the user listend. Like for song X, we sum all the value in column X, then we divided this value by the number of how many songs the user listend, it's matrix.sum(col(X))/matrix.shape[0]

And then we sorted all the average values, and get the top similar list from the data.
