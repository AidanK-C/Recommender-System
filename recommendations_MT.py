'''
CSC381: Building a simple Recommender System

The final code package is a collaborative programming effort between the
CSC381 student(s) named below, the class instructor (Carlos Seminario), and
source code from Programming Collective Intelligence, Segaran 2007.
This code is for academic use/purposes only.

CSC381 Programmer/Researcher: Tasos Pagounas

'''

from audioop import avg
import os
from math import sqrt
import statistics
import numpy as np
import matplotlib.pyplot as plt
import pickle

def from_file_to_dict(path, datafile, itemfile):
    ''' Load user-item matrix from specified file 
        
        Parameters:
        -- path: directory path to datafile and itemfile
        -- datafile: delimited file containing userid, itemid, rating
        -- itemfile: delimited file that maps itemid to item name
        
        Returns:
        -- prefs: a nested dictionary containing item ratings for each user
    
    '''
    
    # Get movie titles, place into movies dictionary indexed by itemID
    movies={}
    try:
        with open (path + '/' + itemfile, encoding='iso8859') as myfile: 
        #with open (path + '/' + itemfile) as myfile: 
            # this encoding is required for some datasets: encoding='iso8859'
            for line in myfile:
                (id,title)=line.split('|')[0:2]
                movies[id]=title.strip()
    
    # Error processing
    except UnicodeDecodeError as ex:
        print (ex)
        print (len(movies), line, id, title)
        return {}
    except Exception as ex:
        print (ex)
        print (len(movies))
        return {}
    
    # Load data into a nested dictionary
    prefs={}
    for line in open(path+'/'+ datafile):
        #print(line, line.split('\t')) #debug
        (user,movieid,rating,ts)=line.split('\t')
        user = user.strip() # remove spaces
        movieid = movieid.strip() # remove spaces
        prefs.setdefault(user,{}) # make it a nested dicitonary
        prefs[user][movies[movieid]]=float(rating)
    
    #return a dictionary of preferences
    return prefs

def data_stats(prefs, filename):
    ''' Computes/prints descriptive analytics:
        -- Total number of users, items, ratings DONE
        -- Overall average rating, standard dev (all users, all items) DONE
        -- Average item rating, standard dev (all users) DONE
        -- Average user rating, standard dev (all items) DONE
        -- Matrix ratings sparsity DONE
        -- Ratings distribution histogram (all users, all items)

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- filename: string containing name of file being analyzed
        
        Returns:
        -- None

    ''' 
    items = []
    total_ratings = 0
    rating_list = []    
    for user in prefs:
        for item in prefs[user]:
            rating_list.append(prefs[user][item])
            if item not in items:
                items.append(item)
            total_ratings += 1
                

    #Total number of users, items, ratings
    print("Total number of users:", len(prefs))   
    print("Total number of rated items:", len(items))
    print("Total number of user-given ratings:", total_ratings)
    
    #Overall average rating, standard dev (all users, all items)
    avg_rating = "{:.2f}".format(sum(rating_list)/len(rating_list))
    std_dev = "{:.2f}".format(statistics.pstdev(rating_list))
    print("Overall average rating " + avg_rating + " out of 5, and std dev of " + std_dev)

    #Average item rating, standard dev (all users)
    items_avg = []
    for item in items:
        tot_critics = 0
        item_score = 0
        for user in prefs:
            if item in prefs[user]:
                tot_critics += 1
                item_score += prefs[user][item]
        items_avg.append(item_score/tot_critics)

    avg_item_rating = "{:.2f}".format(sum(items_avg)/len(items_avg))
    item_std_dev = "{:.2f}".format(statistics.pstdev(items_avg))
    print("Average item rating " + avg_item_rating + " out of 5, and std dev of " + item_std_dev)

    #Average user rating, standard dev (all items)
    users = list(prefs.keys())
    users_avg = []
    for user in users:
        values = prefs[user].values()
        users_avg.append(sum(values)/len(values))
    
    avg_user_rating = "{:.2f}".format(sum(users_avg)/len(users_avg))
    user_std_dev = "{:.2f}".format(statistics.pstdev(users_avg))
    print("Average user rating " + avg_user_rating + " out of 5, and std dev of " + user_std_dev)

    #Matrix ratings sparsity
    sparsity = (1-(total_ratings)/(len(prefs)*(len(items)))) * 100  #calculates sparsity
    str_sparsity = "{:.2f}".format(sparsity) #transforms sparsity to a string with proper decimals
    print("User-Item Matrix Sparsity: " + str_sparsity + "%")

    print()
    print()

    #Ratings distribution histogram (all users, all items)
    all_ratings = []
    for user in prefs:
        for item in prefs[user]:
            all_ratings.append(prefs[user][item])

    hist,bins = np.histogram(all_ratings, bins=[1,2,3,4,5])
    xy = plt.gca()
    xy.set_xlim([1,5])
    xy.set_ylim(0,max(hist))
    xy.set_xticks([1,2,3,4,5])
    plt.hist(all_ratings,bins=[1,2,3,4,5], color="c")
    xy.set_facecolor('white')
    plt.title("Ratings Histogram")
    plt.xlabel("Rating")
    plt.ylabel("Number of user ratings")
    plt.grid()
    plt.show()

def popular_items(prefs, filename):
    ''' Computes/prints popular items analytics    
        -- popular items: most rated (sorted by # ratings) DONE
        -- popular items: highest rated (sorted by avg rating) DONE
        -- popular items: highest rated items that have at least a 
                          "threshold" number of ratings DONE
        
        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- filename: string containing name of file being analyzed
        
        Returns:
        -- None

    '''
    #Creates dictionary with each movie and the times it has been rated
    nr_ratings = {}
    for user in prefs:
        for movie in prefs[user]:
            if movie in nr_ratings:
                nr_ratings[movie] += 1
            else:
                nr_ratings[movie] = 1
    
    sorted_movies = sorted(nr_ratings, key=nr_ratings.get, reverse=True) #sorts movies based on total ratings

    #Creates dictionary with each movies average rating
    movies_rating = {}   
    for movie in nr_ratings:
        score = 0
        count = 0
        for user in prefs:
            if movie in prefs[user]:
               score += prefs[user][movie]
               count += 1 
            else:
                continue
        movies_rating[movie] = score/count

    #Swaps movies with equal ratings amount based on item rating
    for movie in range(len(sorted_movies) - 1):
        for equal in range(movie + 1, len(sorted_movies)):
            if nr_ratings[sorted_movies[movie]] == nr_ratings[sorted_movies[equal]] and movies_rating[sorted_movies[equal]] > movies_rating[sorted_movies[movie]]:
                temp = sorted_movies[movie]
                sorted_movies[movie] = sorted_movies[equal]
                sorted_movies[equal] = temp
    
    
    #Popular items: most rated (sorted by # ratings)
    ITEMS_TO_DISPLAY = 5
    print("Popular Items -- most rated:")
    print()
    print('Title \t\t\t #Ratings \t Avg. Rating')
    
    flag = 0
    for i in sorted_movies:
        flag += 1
        print(i, "\t", nr_ratings[i], "\t\t", "{:.2f}".format(movies_rating[i])) 
        if flag == 5:
            break

    print()
    print()

    print("Popular items -- highest rated")
    print()
    print('Title \t\t\t Avg Rating \t #Ratings')
    sorted_ratings = sorted(movies_rating, key=movies_rating.get, reverse=True) #sorts movies based on highest avg ratings

    flag = 0
    for i in sorted_ratings:
        flag += 1
        print(i, "\t", "{:.2f}".format(movies_rating[i]), "\t\t", nr_ratings[i])  
        if flag == 5:
            break
    
    print()
    print()
    
    print("Overall best rated items (number of ratings >=5)")
    print()
    print('Title \t\t\t Avg Rating \t #Ratings')

    flag = 0
    for i in sorted_ratings:
        if nr_ratings[i] >=5:
            flag += 1
            print(i, "\t", "{:.2f}".format(movies_rating[i]), "\t\t", nr_ratings[i])  
            if flag == 5:
                break
   
def sim_distance(prefs,person1,person2, n = 50):
    '''
        Calculate Euclidean distance similarity 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person1: string containing name of user 1
        -- person2: string containing name of user 2
        
        Returns:
        -- Euclidean distance similarity for RS, as a float
        
    '''
    
    # Get the list of shared_items
    si={}
    for item in prefs[person1]: 
        if item in prefs[person2]: 
            si[item]=1
    
    # if they have no ratings in common, return 0
    if len(si)==0: 
        return 0
    
    factor = 1
    if len(si) < n:
        factor = len(si)/n
    
    # Add up the squares of all the differences
    ## Note: Calculate similarity between any two users across all items they
    ## have rated in common; i.e., includes the sum of the squares of all the
    ## differences
    
    sum_of_squares = 0
    
    ## add code here to calc sum_of_squares ..
    for item in si:
        sum_of_squares += (prefs[person1][item] - prefs[person2][item])**2

    # returns Euclidean distance similarity for RS
    return factor * (1/(1+sqrt(sum_of_squares)))

def sim_pearson(prefs,p1,p2, n = 1):
    '''
        Calculate Pearson Correlation similarity 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person1: string containing name of user 1
        -- person2: string containing name of user 2
        
        Returns:
        -- Pearson Correlation similarity as a float

    '''
    #lists with the respective ratings of each user for shared items
    shared_items1 = []
    shared_items2 = [] 
    for item in prefs[p1]: 
        if item in prefs[p2]: 
            shared_items1.append(prefs[p1][item])
            shared_items2.append(prefs[p2][item])
    
    # if they have no ratings in common, return 0
    if len(shared_items1) == 0: 
        return 0
    
    factor = 1
    if len(shared_items1) < n:
        factor = len(shared_items1)/n
    
    #average rating of each user for all items
    p1_avg = np.mean(shared_items1)
    p2_avg = np.mean(shared_items2)

    numerator = 0
    denominator_p1 = 0
    denominator_p2 = 0

    for item in prefs[p1]: 
        if item in prefs[p2]:
            numerator += ((prefs[p1][item] - p1_avg) * (prefs[p2][item] - p2_avg))
            denominator_p1 += (prefs[p1][item] - p1_avg)**2 
            denominator_p2 += (prefs[p2][item] - p2_avg)**2
    
    if (sqrt(denominator_p1*denominator_p2) == 0):
        return 0
    else:
        return factor * (numerator/(sqrt(denominator_p1*denominator_p2)))

def getRecommendations(prefs,person,similarity=sim_pearson):
    '''
        Calculates recommendations for a given user 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user
        -- similarity: function to calc similarity (sim_pearson is default)
        
        Returns:
        -- A list of recommended items with 0 or more tuples, 
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.
        
    '''
    totals={}
    simSums={}
    for other in prefs:
      # don't compare me to myself
        if other==person: 
            continue
        sim=similarity(prefs,person,other)
    
        # ignore scores of zero or lower
        if sim<=0: continue
        for item in prefs[other]:
            
            # only score movies I haven't seen yet
            if item not in prefs[person] or prefs[person][item]==0:
                # Similarity * Score
                totals.setdefault(item,0)
                totals[item]+=prefs[other][item]*sim
                # Sum of similarities
                simSums.setdefault(item,0)
                simSums[item]+=sim
  
    # Create the normalized list
    rankings=[(total/simSums[item],item) for item,total in totals.items()]
  
    # Return the sorted list
    rankings.sort()
    rankings.reverse()
    return rankings

def getRecommendationsSim(prefs,userMatch,user,thres=0.5): #default threshold
    '''
    Quicker calculation of recommendations for a given user 
    -- prefs: dictionary containing user-item matrix
    -- person: string containing name of user
    -- similarity: function to calc similarity (sim_pearson is default)
    -- thres: similarity threshold
    Returns:
        -- A list of recommended items with 0 or more tuples, 
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.
    '''
    userRatings=prefs[user]
    scores={}
    totalSim={}
    # Loop over items rated by this user
    for (item,rating) in userRatings.items( ):
  
      # Loop over items similar to this one
        for (similarity,item2) in userMatch[item]:
    
            # Ignore if this user has already rated this item
            if item2 in userRatings: continue
            # ignore scores of zero or lower
            if similarity<=0: continue            
            # Weighted sum of rating times similarity
            scores.setdefault(item2,0)
            scores[item2]+=similarity*rating
            # Sum of all the similarities
            totalSim.setdefault(item2,0)
            totalSim[item2]+=similarity
  
    # Divide each total score by total weighting to get an average

    rankings=[(score/totalSim[item],item) for item,score in scores.items( )]    
  
    # Return the rankings from highest to lowest
    rankings.sort( )
    rankings.reverse( )
    return rankings
def get_all_UU_recs(prefs, sim=sim_pearson, num_users=10, top_N=5):
   ''' 
   Print user-based CF recommendations for all users in dataset

   Parameters
   -- prefs: nested dictionary containing a U-I matrix
   -- sim: similarity function to use (default = sim_pearson)
   -- num_users: max number of users to print (default = 10)
   -- top_N: max number of recommendations to print per user (default = 5)

   Returns: None
   '''
   if sim == sim_pearson:
       print("Using sim_pearson:")

       for user in prefs:
           print ('User-based CF recs for %s: ' % (user), getRecommendations(prefs, user))
   else:
       print()
       print("Using sim_distance:")

       for user in prefs:
           print('User-based CF recs for %s: ' % (user), getRecommendations(prefs, user, similarity=sim_distance))

def loo_cv(prefs, metric, sim, algo):
    """
    Leave_One_Out Evaluation: evaluates recommender system ACCURACY
     
     Parameters:
         prefs dataset: critics, ml-100K, etc.
         metric: MSE, MAE, RMSE, etc.
         sim: distance, pearson, etc.
         algo: user-based recommender, item-based recommender, etc.
     
    Returns:
         error_total: MSE, MAE, RMSE totals for this set of conditions
         error_list: list of actual-predicted differences
    
    
    Algo Pseudocode ..
    Create a temp copy of prefs
    
    For each user in temp copy of prefs:
      for each item in each user's profile:
          delete this item
          get recommendation (aka prediction) list
          restore this item
          if there is a recommendation for this item in the list returned
              calc error, save into error list
          otherwise, continue
      
    return mean error, error list
    """        
    error_list = []
    prefs_copy = prefs.copy()
    for user in prefs_copy:
        for item in list(prefs_copy[user].keys()):
            item_del = prefs[user][item]
            del prefs_copy[user][item]
            recc_list = getRecommendations(prefs_copy, user, similarity = sim)     
            prefs_copy[user][item] = item_del

            for recc in recc_list:
                if item in recc:
                    if item == recc[1]:
                        err = (recc[0]-item_del)**2
                        print('User: %s, Item: %s, Prediction: %.5f, Actual: %.5f, Error: %.5f'% (user, item, \
                                    recc[0], item_del, err))
                        error_list.append(err)
        
                
    error_total = np.mean(error_list)
    
    if metric == "MSE":
        return error_total , error_list

def getRecommendedItems(prefs,itemMatch,user,thres=0.5):
    '''
        Calculates recommendations for a given user 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user
        -- similarity: function to calc similarity (sim_pearson is default)
        
        Returns:
        -- A list of recommended items with 0 or more tuples, 
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.
        
    '''    
    userRatings=prefs[user]
    scores={}
    totalSim={}
    # Loop over items rated by this user
    for (item,rating) in userRatings.items( ):
  
      # Loop over items similar to this one
        for (similarity,item2) in itemMatch[item]:
    
            # Ignore if this user has already rated this item
            if item2 in userRatings: continue
            # ignore scores of zero or lower
            if similarity<=thres: continue            
            # Weighted sum of rating times similarity
            scores.setdefault(item2,0)
            scores[item2]+=similarity*rating
            # Sum of all the similarities
            totalSim.setdefault(item2,0)
            totalSim[item2]+=similarity
  
    # Divide each total score by total weighting to get an average

    rankings=[(score/totalSim[item],item) for item,score in scores.items( )]    
  
    # Return the rankings from highest to lowest
    rankings.sort( )
    rankings.reverse( )
    return rankings

def topMatches(prefs,person,similarity=sim_pearson, n=5):
    '''
        Returns the best matches for person from the prefs dictionary

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user
        -- similarity: function to calc similarity (sim_pearson is default)
        -- n: number of matches to find/return (5 is default)
        
        Returns:
        -- A list of similar matches with 0 or more tuples, 
           each tuple contains (similarity, item name).
           List is sorted, high to low, by similarity.
           An empty list is returned when no matches have been calc'd.
        
    '''     
    scores=[(similarity(prefs,person,other),other) 
                    for other in prefs if other!=person]
    scores.sort()
    scores.reverse()
    return scores[0:n]

def transformPrefs(prefs):
    '''
        Transposes U-I matrix (prefs dictionary) 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        
        Returns:
        -- A transposed U-I matrix, i.e., if prefs was a U-I matrix, 
           this function returns an I-U matrix
        
    '''     
    result={}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item,{})
            # Flip item and person
            result[item][person]=prefs[person][item]
    return result

def calculateSimilarItems(prefs,n=10,similarity=sim_pearson):

    '''
        Creates a dictionary of items showing which other items they are most 
        similar to. 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- n: number of similar matches for topMatches() to return
        -- similarity: function to calc similarity (sim_pearson is default)
        
        Returns:
        -- A dictionary with a similarity matrix
        
    '''     
    result={}
    # Invert the preference matrix to be item-centric
    itemPrefs=transformPrefs(prefs)
    c=0
    for item in itemPrefs:
      # Status updates for larger datasets
        c+=1
        if c%100==0: 
            print ("%d / %d" % (c,len(itemPrefs)))
            
        # Find the most similar items to this one
        scores=topMatches(itemPrefs,item,similarity,n=n)
        result[item]=scores
    return result

def calculateSimilarUsers(prefs,n=100,similarity=sim_pearson):

    '''
        Creates a dictionary of items showing which other users they are most 
        similar to. 

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- n: number of similar matches for topMatches() to return
        -- similarity: function to calc similarity (sim_pearson is default)
        
        Returns:
        -- A dictionary with a similarity matrix
        
    '''     
    result={}
    # Invert the preference matrix to be item-centric
    
    c=0
    for user in prefs:
      # Status updates for larger datasets
        c+=1
        if c%100==0: 
            print ("%d / %d" % (c,len(prefs)))
            
        # Find the most similar items to this one
        scores=topMatches(prefs,user,similarity,n=n)
        result[user]=scores
    
    return result

def get_all_II_recs(prefs, itemsim, sim_method, num_users=10, top_N=5):
    ''' 
    Print item-based CF recommendations for all users in dataset

    Parameters
    -- prefs: U-I matrix (nested dictionary)
    -- itemsim: item-item similarity matrix (nested dictionary)
    -- sim_method: name of similarity method used to calc sim matrix (string)
    -- num_users: max number of users to print (integer, default = 10)
    -- top_N: max number of recommendations to print per user (integer, default = 5)

    Returns: None
    
    '''
    for user in prefs:
        print ('Item-based CF recs for %s, %s: ' % (user, sim_method), getRecommendedItems(prefs, itemsim, user))

def loo_cv_sim(prefs, sim, algo, sim_matrix):
    """
    Leave-One_Out Evaluation: evaluates recommender system ACCURACY
     
     Parameters:
         prefs dataset: critics, etc.
         metric: MSE, or MAE, or RMSE
         sim: distance, pearson, etc.
         algo: user-based recommender, item-based recommender, etc.
         sim_matrix: pre-computed similarity matrix
	 
    Returns:
         error_total: MSE, or MAE, or RMSE totals for this set of conditions
	     error_list: list of actual-predicted differences
    """
    error_list = []
    abs_error_list = []
    prefs_copy = prefs.copy()

        
    for user in prefs:
        for item in list(prefs_copy[user].keys()):
            item_del = prefs[user][item]
            del prefs_copy[user][item]

            if algo == getRecommendedItems:
                recc_list = getRecommendedItems(prefs_copy,sim_matrix,user)
            if algo == getRecommendationsSim:
                recc_list = getRecommendationsSim(prefs_copy,sim_matrix,user)

            prefs_copy[user][item] = item_del

               
            for recc in recc_list:
                if item in recc:
                    if item == recc[1]:
                        err = (recc[0]-item_del)**2
                        err_mae = abs(recc[0]-item_del)
                        print('User: %s, Item: %s, Prediction: %.5f, Actual: %.5f, Error: %.5f'% (user, item, \
                                        recc[0], item_del, err))                          
                        error_list.append(err)
                        abs_error_list.append(err_mae)

    error_total = np.mean(error_list)

    #MSE stats
    MSE = error_total
    MSE_err_list = error_list

    #MAE stats
    MAE = np.mean(abs_error_list)
    MAE_err_list = abs_error_list

    #RMSE stats
    RMSE = sqrt(error_total)
    RMSE_err_list = error_list

    return MSE, MSE_err_list, MAE, MAE_err_list, RMSE, RMSE_err_list
       
def main():
    ''' User interface for Python console '''
    
    # Load critics dict from file
    path = os.getcwd() # this gets the current working directory
                       # you can customize path for your own computer here
    print('\npath: %s' % path) # debug
    done = False
    prefs = {}
    itemsim = {}
    
    while not done: 
        print()
        # Start a simple dialog
        file_io = input('R(ead) critics data from file?, \n'
                        'P(rint) the U-I matrix?, \n'
                        'V(alidate) the dictionary?, \n'
                        'S(tats) print? \n'
                        'D(istance) critics data? \n' 
                        'PC(earson Correlation) critics data? \n'
                        'U(ser-based CF Recommendations)? \n'
                        'LCV(eave one out cross-validation)? \n'
                        'LCVSIM(eave one out cross-validation)? \n'
                        'Sim(ilarity) matrix calc? \n'
                        'SIMU(ilarity matrix user) calc?\n'
                        'I(tem-based CF Recommendations)?  \n'
                        'RML(ead ml100K data)?, \n==>')
        
        if file_io == 'R' or file_io == 'r':
            print()
            file_dir = 'data/'
            datafile = 'critics_ratings.data'
            itemfile = 'critics_movies.item'
            print ('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir + datafile, file_dir + itemfile)
            print('Number of users: %d\nList of users:' % len(prefs), 
                  list(prefs.keys()))

        elif file_io == 'RML' or file_io == 'rml':
            print()
            file_dir = 'data/ml-100k/' # path from current directory
            datafile = 'u.data'  # ratings file
            itemfile = 'u.item'  # movie titles file            
            print ('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir+datafile, file_dir+itemfile)
            print('Number of users: %d\nList of users [0:10]:' 
                      % len(prefs), list(prefs.keys())[0:10] )
            
        elif file_io == 'P' or file_io == 'p':
            # print the u-i matrix
            print()
            if len(prefs) > 0:
                print ('Printing "%s" dictionary from file' % datafile)
                print ('User-item matrix contents: user, item, rating')
                for user in prefs:
                    for item in prefs[user]:
                        print(user, item, prefs[user][item])
            else:
                print ('Empty dictionary, R(ead) in some data!')
                
        elif file_io == 'V' or file_io == 'v':      
            print()
            if len(prefs) > 0:
                # Validate the dictionary contents ..
                print ('Validating "%s" dictionary from file' % datafile)
                print ("critics['Lisa']['Lady in the Water'] =", 
                       prefs['Lisa']['Lady in the Water']) # ==> 2.5
                print ("critics['Toby']:", prefs['Toby']) 
                # ==> {'Snakes on a Plane': 4.5, 'You, Me and Dupree': 1.0, 
                #      'Superman Returns': 4.0}
            else:
                print ('Empty dictionary, R(ead) in some data!')
                
        elif file_io == 'S' or file_io == 's':
            print()
            filename = 'critics_ratings.data'
            if len(prefs) > 0:
                data_stats(prefs, filename)
                popular_items(prefs, filename)
            else: # Make sure there is data  to process ..
                print ('Empty dictionary, R(ead) in some data!')   
        
        elif file_io == 'D' or file_io == 'd':
            print()
            if len(prefs) > 0:                            
                print('User-User distance similarities:')
            

                users = list(prefs.keys())
                for user1 in range(len(users) - 1):
                    for user2 in range((user1 + 1), len(users)):
                        print ('Distance similarity between ' + users[user1] + ' and ' + users[user2] + ': ' + str(sim_distance(prefs, users[user1], users[user2])))
                
                print()
             
            else:
                print ('Empty dictionary, R(ead) in some data!')   
         
        elif file_io == 'PC' or file_io == 'pc':
            print()
            if len(prefs) > 0:             
                print('Pearson for all users:')
                # Calc Pearson for all users
                
                ## add some code here to calc User-User Pearson Correlation similarities
                users = list(prefs.keys())
                for user1 in range(len(users) - 1):
                    for user2 in range((user1 + 1), len(users)):
                        print ('Pearson similarity between ' + users[user1] + ' and ' + users[user2] + ': ' + str(sim_pearson(prefs, users[user1], users[user2]))) 
                print()
                
            else:
                print ('Empty dictionary, R(ead) in some data!')

        elif file_io == 'U' or file_io == 'u':
            print()
            if len(prefs) > 0:             
                print ('Example:')
                user_name = 'Toby'
                print ('User-based CF recs for %s, sim_pearson: ' % (user_name), 
                       getRecommendations(prefs, user_name)) 

                print ('User-based CF recs for %s, sim_distance: ' % (user_name),
                       getRecommendations(prefs, user_name, similarity=sim_distance)) 

                print()
                
                print('User-based CF recommendations for all users:')
                # Calc User-based CF recommendations for all users
        
                get_all_UU_recs(prefs, sim=sim_pearson, num_users = 10, top_N = 5)
                get_all_UU_recs(prefs, sim=sim_distance, num_users = 10, top_N = 5)

                print()
                
            else:
                print ('Empty dictionary, R(ead) in some data!')           

        elif file_io == 'LCV' or file_io == 'lcv':
            print()
            if len(prefs) > 0:             
                print ('LOO_CV Evaluation:')            
                ## add some code here to calc LOOCV 
                ## write a new function to do this ..
                sim = sim_pearson
                algo = getRecommendations
                # MSE, MSE_list = loo_cv(prefs,'MSE', sim, algo)
                MSE, MSE_list = loo_cv(prefs,'MSE', sim, algo)
                print('MSE for critics: %.10f, using' %MSE, sim)
                print()
                sim = sim_distance
                MSE, MSE_list = loo_cv(prefs,'MSE', sim, algo)
                print('MSE for critics: %.10f, using' %MSE, sim)
            else:
                print ('Empty dictionary, R(ead) in some data!')            

        elif file_io == 'Sim' or file_io == 'sim':
            print()
            if len(prefs) > 0: 
                ready = False # sub command in progress
                sub_cmd = input('RD(ead) distance or RP(ead) pearson or WD(rite) distance or WP(rite) pearson? ')
                try:
                    if sub_cmd == 'RD' or sub_cmd == 'rd':
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(open("save_itemsim_distance.p", "rb" ))
                        sim_method = 'sim_distance'
    
                    elif sub_cmd == 'RP' or sub_cmd == 'rp':
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(open( "save_itemsim_pearson.p", "rb" ))  
                        sim_method = 'sim_pearson'
                        
                    elif sub_cmd == 'WD' or sub_cmd == 'wd':
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(prefs,similarity=sim_distance)                     
                        # Dump/save dictionary to a pickle file
                        pickle.dump(itemsim, open( "save_itemsim_distance.p", "wb" ))
                        sim_method = 'sim_distance'
                        
                    elif sub_cmd == 'WP' or sub_cmd == 'wp':
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(prefs,similarity=sim_pearson)                     
                        # Dump/save dictionary to a pickle file
                        pickle.dump(itemsim, open( "save_itemsim_pearson.p", "wb" )) 
                        sim_method = 'sim_pearson'
                    
                    else:
                        print("Sim sub-command %s is invalid, try again" % sub_cmd)
                        continue
                    
                    ready = True # sub command completed successfully
                    
                except Exception as ex:
                    print ('Error!!', ex, '\nNeed to W(rite) a file before you can R(ead) it!'
                           ' Enter Sim(ilarity matrix) again and choose a Write command')
                    print()
                

                if len(itemsim) > 0 and ready == True: 
                    # Only want to print if sub command completed successfully
                    print ('Similarity matrix based on %s, len = %d' 
                           % (sim_method, len(itemsim)))
                    print()
                    if (sim_method == 'sim_distance'):
                        result = calculateSimilarItems(prefs, n = 100, similarity = sim_distance)
                    else:
                        result = calculateSimilarItems(prefs)
                    for item in result:
                        print(item, result[item])                    

                print()
                
            else:
                print ('Empty dictionary, R(ead) in some data!')

        elif file_io == 'SIMU' or file_io == 'simu':
            print()
            if len(prefs) > 0: 
                ready = False # sub command in progress
                sub_cmd = input('RD(ead) distance or RP(ead) pearson or WD(rite) distance or WP(rite) pearson?')
                try:
                    if sub_cmd == 'RD' or sub_cmd == 'rd':
                        # Load the dictionary back from the pickle file.
                        usersim = pickle.load(open( "save_usersim_distance.p", "rb" ))
                        sim_method = 'sim_distance'
    
                    elif sub_cmd == 'RP' or sub_cmd == 'rp':
                        # Load the dictionary back from the pickle file.
                        usersim = pickle.load(open( "save_usersim_pearson.p", "rb" ))  
                        sim_method = 'sim_pearson'
                    

                    elif sub_cmd == 'WD' or sub_cmd == 'wd':
                        
                        usersim = calculateSimilarUsers(prefs,similarity=sim_distance)                     
                        # Dump/save dictionary to a pickle file
                        pickle.dump(usersim, open( "save_usersim_distance.p", "wb" ))
                        sim_method = 'sim_distance'
                        
                    elif sub_cmd == 'WP' or sub_cmd == 'wp':
                        
                        usersim = calculateSimilarUsers(prefs,similarity=sim_pearson)                     
                        # Dump/save dictionary to a pickle file
                        pickle.dump(usersim, open( "save_usersim_pearson.p", "wb" )) 
                        sim_method = 'sim_pearson'
                    
                  
                    else:
                        print("Sim sub-command %s is invalid, try again" % sub_cmd)
                        continue
                    
                    ready = True # sub command completed successfully
                    
                except Exception as ex:
                    print ('Error!!', ex, '\nNeed to W(rite) a file before you can R(ead) it!'
                           ' Enter Sim(ilarity matrix) again and choose a Write command')
                    print()
                if len(usersim) > 0 and ready == True: 
                # Only want to print if sub command completed successfully
                    print('Similarity matrix based on %s, len = %d' 
                        % (sim_method, len(usersim)))
                    print()
                    ##
                    ## enter new code here, or call a new function, 
                    if (sim_method=='sim_distance'):
                        result = calculateSimilarUsers(prefs, n=100, similarity=sim_distance)
                 
                    else:
                        result = calculateSimilarUsers(prefs)
                    for user in result:
                        print(user, result[user])
                print()

        elif file_io == 'LCVSIM' or file_io == 'lcvsim':
            print()
            file_io = input('Enter U(ser) or I(tem) algo:')
            if file_io == 'U' or file_io == 'u':
                if len(prefs) > 0 and usersim !={}:             
                    print('LOO_CV_SIM Evaluation')
                   
                    prefs_name = 'critics' ###MUST BE ML-100k
                   
                    algo = getRecommendationsSim #user-based

                    if sim_method == 'sim_pearson': 
                        sim = sim_pearson
                        MSE, MSE_list, MAE, MAE_list, RMSE,RMSE_list = loo_cv_sim(prefs, sim, algo, usersim)
                        print('MSE for %s: %.5f, len(MSE list): %d, MAE: %.5f, len(MAE list): %d,\
                            RMSE: %.5f, len(RMSE list): %d, using %s' %(prefs_name, MSE, len(MSE_list),MAE,len(MAE_list),RMSE, len(RMSE_list), sim))
                        print()
                    
                    if sim_method == 'sim_distance':
                        sim = sim_distance
                        MSE,MSE_list, MAE, MAE_list, RMSE,RMSE_list = loo_cv_sim(prefs, sim, algo, usersim)
                        print('MSE for %s:%.5f, len(MSE list): %d, MAE: %.5f, len(MAE list): %d,\
                            RMSE: %.5f, len(RMSE list): %d, using %s' %(prefs_name,MSE, len(MSE_list),MAE,len(MAE_list),RMSE, len(RMSE_list), sim))
                 
                    else:
                        print('Run Sim(ilarity matrix) command to create/load Sim matrix!')
                
                else:
                    print ('Empty dictionary, run RML(ead ml100K) OR Empty Sim Matrix, run Sim!')
            elif file_io == 'I' or file_io == 'i':    
                if len(prefs) > 0 and itemsim !={}:             
                    print('LOO_CV_SIM Evaluation')
                   
                    prefs_name = 'critics' ###AGAIN CHANGE TO ML-100k

                    algo = getRecommendedItems ## Item-based recommendation
                    
                    
                    if sim_method == 'sim_pearson': 
                        sim = sim_pearson
                        MSE, MSE_list, MAE, MAE_list, RMSE,RMSE_list = loo_cv_sim(prefs, sim, algo, itemsim)
                        print('MSE for %s: %.5f, len(MSE list): %d, MAE: %.5f, len(MAE list): %d,\
                            RMSE: %.5f, len(RMSE list): %d, using %s' %(prefs_name, MSE, len(MSE_list),MAE,len(MAE_list),RMSE, len(RMSE_list), sim))
                        print()
                    
                    if sim_method == 'sim_distance':
                        sim = sim_distance
                        MSE, MSE_list, MAE, MAE_list, RMSE,RMSE_list = loo_cv_sim(prefs, sim, algo, itemsim)
                        print('MSE for %s: %.5f, len(MSE list): %d, MAE: %.5f, len(MAE list): %d,\
                            RMSE: %.5f, len(RMSE list): %d, using %s' %(prefs_name, MSE, len(MSE_list),MAE,len(MAE_list),RMSE, len(RMSE_list), sim))
                        print()
               
                    else:
                        print('Run Sim(ilarity matrix) command to create/load Sim matrix!')
                 
                else:
                    print ('Empty dictionary, run RML(ead ml100K) OR Empty Sim Matrix, run Sim!')  
                              
        else:
            done = True
    
    print('\nGoodbye!')
        
if __name__ == '__main__':
    main()