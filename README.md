# MusicRecommendation

This music recommendation task concentrates on two different ways for recommendation:
1. **Popularity-based**, which is used for new user who didn't have music listened record 
2. **Item-based collabrative filtering**, which is used for regular user who has music listened record

## Data Exploration
This instruction focuses on how to make recommendation, for the topic that how to explore data please refer to the code file.
The structure of dataframe which is used in the recommendation shows as below:

## Popularity-base Recommendation
Using a top list for recommendation for new user, because those users don't have listened record so that can't apply collabrative filtering on recommendation for them.
''' python
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
  '''
