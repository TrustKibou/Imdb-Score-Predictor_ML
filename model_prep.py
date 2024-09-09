import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import tree
from math import sqrt
import matplotlib.pyplot as plt
import os
import numpy as np



# FOR USE IN MAIN ... had to do this in order to use these vars in main, while also showing the loading_frame while model was training.
class Model:
    def __init__(self):
        self.movie_data = None
        self.model = None
        self.director_name_count = None
        self.actor1_name_count = None
        self.actor2_name_count = None
        self.actor3_name_count = None
        self.model_stats = None
        self.y_actual = None
        self.y_predicted = None


##################################################################################################
################################################################################# MODEL STATISTICS
##################################################################################################
def print_model_eval(modelName, y_actual, y_predicted, entryTotal, featureTotal):
    mse = mean_squared_error(y_actual, y_predicted)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_actual, y_predicted)
    mape = np.mean(np.abs(y_actual - y_predicted)) * 100
    r2 = r2_score(y_actual, y_predicted)
    r2adj = 1 - ((1 - r2) * (entryTotal - 1) / (entryTotal - featureTotal - 1))

    print(modelName, "PERFORMANCE—————————————————————————————————————————————————————————————")
    print("Mean Squared Error: ", mse)
    print("Root Mean Squared Error: ", rmse)
    print("Mean Absolute Error: ", mae)
    print("Mean Absolute Percentage Error: ", mape)
    print("R-Squared: ", r2)
    print("Adjusted R-Squared: ", r2adj)



def get_model_eval(y_actual, y_predicted, entryTotal, featureTotal):
    model_stats = {}
    model_stats["mse"] = mean_squared_error(y_actual, y_predicted)
    model_stats["rmse"] = sqrt(model_stats["mse"])
    model_stats["mae"] = mean_absolute_error(y_actual, y_predicted)
    model_stats["mape"] = np.mean(np.abs(y_actual - y_predicted)) * 100
    model_stats["r2"] = r2_score(y_actual, y_predicted)
    model_stats["r2adj"] = 1 - ((1 - model_stats["r2"]) * (entryTotal - 1) / (entryTotal - featureTotal - 1))

    return model_stats


##################################################################################################
#################################################################################### MODEL FIGURES
##################################################################################################

def create_figures(y_actual, y_predicted, modelName):

    figures_list = []

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.scatter(y_actual, y_predicted, alpha=0.2)
    ax1.set(title=modelName + " - Predicted vs. Actual Scores", xlabel="Actual IMDb Scores", ylabel="Predicted IMDb Scores")
    ax1.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], color="black")

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.scatter(y_predicted, y_actual - y_predicted, alpha=0.2)
    ax2.set(title=modelName + " - Error Value per Prediction", xlabel="Predicted IMDb Scores", ylabel="Error Amount")
    ax2.axhline(y=0, lw=1, color="red", linestyle="--")

    figures_list.append(fig1)
    figures_list.append(fig2)

    plt.close("all")

    return figures_list



    # ### ACTUAL VS. PREDICTED
    # plot1 = plt.figure(figsize=(15,10))
    # plt.scatter(y_actual, y_predicted, alpha=0.2)       #NOTE: alpha=transparency
    # plt.title(modelName + " - Actual vs. Predicted")
    # plt.xlabel("Actual IMDb Scores")
    # plt.ylabel("Predicted IMDb Scores")
    # plt.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], color="black")
    # plt.savefig(dir_figures + plotURL + "predVactual.png", dpi=300) # save to drive
    # # plt.show()
    # plt.close()
    #
    # ### ERROR RATE PER PREDICTION
    # plt.figure(figsize=(15,10))
    # plt.scatter(y_predicted, y_actual - y_predicted)
    # plt.title(modelName + " - Error Value per Prediction")
    # plt.xlabel("Predicted IMDb Scores")
    # plt.ylabel("Error Total")
    # plt.axhline(y=0, lw=1, color="red", linestyle="--")
    # plt.savefig(dir_figures + plotURL + "errorVpred", dpi=300) # save to drive
    # # plt.show()
    # plt.close()



##################################################################################################
############################################################################### READ DATA FROM CSV
##################################################################################################

def preprocessAndTrain(model):

    ###### GET DATASET FROM CSV
    movie_data = pd.read_csv("datasets/movies_training.csv")

    ##### CREATE FIGURES DIRECTORY
    dir_figures = "model_figures"

    if not os.path.exists(dir_figures):
        try:
            os.makedirs(dir_figures)
        except OSError as err:
            print("Error: ", err)

    ##################################################################################################
    ################################################################################ INITIAL DATA VIEW
    ##################################################################################################
    # view settings
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_info_columns", 500)
    #
    #
    # # LEAD ROWS
    # print(">>> DATA PREVIEW (FIRST SET OF ROWS)—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————")
    # print(movie_data.head(), end="\n\n\n")
    #
    # # SPECS (cols, NaN count, data type)
    # print(">>> DATA SPECS—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————")
    # print(movie_data.info(), end="\n\n\n")
    #
    # # ROWS x COLS
    # print(">>> (ROWS, COLS)—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————")
    # print(movie_data.shape, end="\n\n\n")
    #
    # # STATISTICAL
    # print(">>> STATISTICS—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————")
    # print(movie_data.describe(), end="\n\n\n")
    #
    # # VIEW NULL VALS per COL
    # print(">>> TOTAL NULL per COLUMN—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————")
    # print(movie_data.isnull().sum(), end="\n\n\n")



    ##################################################################################################
    #################################################################### DATA CLEAN - NULL, TYPE, DUPE
    ##################################################################################################

    ##########################
    ######### PREP NULL VALUES
    ##########################                                          NOTES: inplace=True modifies original DF instead of creating new one ... watch fillna syntax

    ### DROPS------------vars are too important to risk estimations
    # budget & gross - removes 20% ... (b-492, g-884)
    movie_data = movie_data.dropna(axis = 0, subset = ["budget"] )
    movie_data = movie_data.dropna(axis = 0, subset = ["gross"] )


    ### EXPLICIT REPLACEMENT------------ evaluate importance during analysis
    # director_name———104       # actor_1_name———7      # actor_2_name———13     # actor_3_name———23     # plot_keywords———153
    movie_data.fillna({"director_name": "unknown"}, inplace = True)
    movie_data.fillna({"actor_1_name": "unknown"}, inplace = True)
    movie_data.fillna({"actor_2_name": "unknown"}, inplace = True)
    movie_data.fillna({"actor_3_name": "unknown"}, inplace = True)
    movie_data["plot_keywords"] = movie_data["plot_keywords"].fillna(movie_data["genres"].str.lower()) # avoid inplace due to futurewarning

    # language———2          # content_rating———303
    movie_data.fillna({"language": "english"}, inplace = True)
    movie_data.fillna({"content_rating": "unrated"}, inplace = True)

    # repeat content_rating values - dimensionality reduction
    # print(movie_data["content_rating"].value_counts())
    movie_data["content_rating"] = movie_data["content_rating"].replace("Approved", "PG")   # ESTIMATE - review impact on data/preds
    movie_data["content_rating"] = movie_data["content_rating"].replace("X", "NC-17")       # changed in 1990 (not associated with porn)
    movie_data["content_rating"] = movie_data["content_rating"].replace("Passed", "PG")     # ESTIMATE - review impact on data/preds
    movie_data["content_rating"] = movie_data["content_rating"].replace("M", "PG")          # changed to GP, which also changed to PG
    movie_data["content_rating"] = movie_data["content_rating"].replace("GP", "PG")         # changed in 1972
    movie_data["content_rating"] = movie_data["content_rating"].replace("Not Rated", "Unrated")         # generally the same thing
    # print(movie_data["content_rating"].value_counts())



    ### AVERAGES------------
    # director_facebook_likes———104         # actor_1_facebook_likes———7       # actor_2_facebook_likes———13        # actor_3_facebook_likes———23
    # facenumber_in_poster———13             # duration———15                    # num_critic_for_reviews———50        # num_user_for_reviews———21
    movie_data.fillna({"director_facebook_likes": movie_data["director_facebook_likes"].mean()}, inplace = True)
    movie_data.fillna({"actor_1_facebook_likes": movie_data["actor_1_facebook_likes"].mean()}, inplace = True)
    movie_data.fillna({"actor_2_facebook_likes": movie_data["actor_2_facebook_likes"].mean()}, inplace = True)
    movie_data.fillna({"actor_3_facebook_likes": movie_data["actor_3_facebook_likes"].mean()}, inplace = True)
    movie_data.fillna({"facenumber_in_poster": movie_data["facenumber_in_poster"].mean()}, inplace = True)
    movie_data.fillna({"duration": movie_data["duration"].mean()}, inplace = True)
    movie_data.fillna({"num_critic_for_reviews": movie_data["num_critic_for_reviews"].mean()}, inplace = True)
    movie_data.fillna({"num_user_for_reviews": movie_data["num_user_for_reviews"].mean()}, inplace = True)

    ### MOST COMMON------------ (eval"d in excel)
    # color———19        # aspect_ratio———329
    movie_data.fillna({"color": "color"}, inplace = True)
    movie_data.fillna({"aspect_ratio": movie_data.aspect_ratio.mode().iloc[0]}, inplace = True)       # returns series; get first index [most common]

    # # REVIEW NULL VALUES ---------------------
    # print(">>> TOTAL NULL AFTER REPLACEMENTS—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————")
    # print(movie_data.isnull().sum(), end="\n\n\n")


    ##########################
    ######## REMOVE DUPLICATES
    ##########################
    movie_data.drop_duplicates(inplace=True)        # .009% dropped [45 ents]



    ##########################
    ####### CONVERT DATA TYPES
    ##########################
    ### FLOAT--->INT
    movie_data["title_year"] = movie_data["title_year"].astype(int)                                 # title_year
    movie_data["duration"] = movie_data["duration"].astype(int)                                     # duration
    movie_data["budget"] = movie_data["budget"].astype(int)                                         # budget
    movie_data["gross"] = movie_data["gross"].astype(int)                                           # gross
    movie_data["director_facebook_likes"] = movie_data["director_facebook_likes"].astype(int)       # director_facebook_likes
    movie_data["actor_1_facebook_likes"] = movie_data["actor_1_facebook_likes"].astype(int)         # actor_1_facebook_likes
    movie_data["actor_2_facebook_likes"] = movie_data["actor_2_facebook_likes"].astype(int)         # actor_2_facebook_likes
    movie_data["actor_3_facebook_likes"] = movie_data["actor_3_facebook_likes"].astype(int)         # actor_3_facebook_likes
    movie_data["facenumber_in_poster"] = movie_data["facenumber_in_poster"].astype(int)             # facenumber_in_poster
    movie_data["num_critic_for_reviews"] = movie_data["num_critic_for_reviews"].astype(int)         # num_critic_for_reviews
    movie_data["num_user_for_reviews"] = movie_data["num_user_for_reviews"].astype(int)             # num_user_for_reviews

    # VERIFY SPECS
    # print(">>> DATA SPECS AFTER TYPE CONVERSION—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————")
    # print(movie_data.info(), end="\n\n\n")


    ##########################
    ####### FEATURE ADJUSTMENT
    ##########################

    ### REMOVE --> IMDB LINK, MOVIE TITLE, LANGUAGE (due to majority being English)
    movie_data = movie_data.drop(columns=["movie_imdb_link", "movie_title", "language"])


    ### ADD --> PROFIT
    movie_data["profit"] = movie_data["gross"] - movie_data["budget"]


    # REVIEW NEW SPECS AFTER COL ADJUSTMENT
    # print(">>> DATA SPECS—————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————")
    # print(movie_data.info(), end="\n\n\n")


    ##################################################################################################
    ######################################################################## DATA CLEAN - VALUE REPAIR
    ##################################################################################################
                                                                                                ###### MISSING VALUES: LANGUAGE, COUNTRY ........ PERFORMED MANUALLY IN EXCEL
                                                                                                ###### UNKNOWN SYMBOLS IN VALUES - PERFORMED MANUALLY IN EXCEL (5,298 CELLS REPAIRED)

    ##########################
    ############### WHITESPACE & LOWER
    ##########################
    for column in movie_data.select_dtypes(include=["object"]).columns:
        movie_data[column] = movie_data[column].str.strip()
        movie_data[column] = movie_data[column].str.lower()



    ##########################
    # VALUE SPLIT -> MULT COLS
    ##########################

    ##########
    ### GENRES
    ##########
    # check if grouping possible
    # print(movie_data["genres"].value_counts()) # TOO COMPLICATED DUE TO MULTI-WORD VALUES

    # create alt data frame
    genres = movie_data["genres"].str.get_dummies(sep="|")                                    ####NOTE: get_dummies creates new df based on vals (((in this case, vals of a col separated by |)))

    # prepend "<coltitle>_" to each col name
    genres.columns = ["genre_" + column for column in genres.columns]

    # merge alt data frames with movie data frame
    movie_data = pd.concat([movie_data, genres], axis=1)                                 ####NOTE: axis=1 refers to col ops [0=rows]

    # remove original genres/plot_keyword cols (avoid redundancy duuuuuuuuuh)
    movie_data.drop("genres", axis=1, inplace=True)                                     ####NOTE: inplace decides whether to create new df (False) or modify original (False)



    movie_data.drop(columns = "plot_keywords", axis=1, inplace=True) # DECIDED TO DROP FOR TEMPORARY EASE OF USER DATA
    # ########### --- old
    # # PLOT_KEY
    # ###########                      !!! PROBLEM: splitting plot_keywords into a boolean col per val would result in an addition 6,800+ columns; NOT PRACTICAL
    #                                             # instead, let"s perform a lil" bit of DiMeNsIoNaLiTy ReDuCtIoN!!!!
    #                                             # NOTE: instead of modifying the frame within the (future) series loop, we instead create a series for each column and then use pd.concat... this allows us to achieve our goal without causing performance risks (according to pandas)
    #
    # # CREATE LIST OF PLOT_KEYS
    # keywordList = movie_data["plot_keywords"].str.split("|").explode()
    # # ................................................................... returns len( Series ) = 5,042
    #
    # # CALCULATE KEYWORD FREQUENCY TO DETERMINE VALUABLE FEATURES
    # keyCount = keywordList.value_counts()
    # # ................................................................... RESULTS: MAJORITY <10 ... KEEP TOP 200 (range of 14 - 153 occurences)
    #
    # # RETAIN HIGH FREQUENCY KEYWORDS
    # topKeywords = keyCount.head(200).index.tolist()
    # # print(keyCount.head(200))
    #
    # # CREATE SERIES LIST TO LATER MERGE
    # series_list = []
    #
    # # LOOP THROUGH EACH KEYWORD AND CREATE A SERIES THAT"S VALS INDICATE HOW MANY TIMES THE KEY OCCURS FOR THAT ROW
    # for key in topKeywords:
    #     colName = "plotKey_" + key      # col name
    #     keyseries = pd.Series(movie_data["plot_keywords"].str.contains(key, na=False).astype(int), name=colName)
    #     series_list.append(keyseries)
    #
    # # COMBINE TEMP DF WITH MOVIE DF
    # movie_data = pd.concat([movie_data] + series_list, axis=1)
    #
    # # DROP ORIGINAL PLOT_KEYWORDS COLUMN
    # movie_data.drop("plot_keywords", axis=1, inplace=True)
    #
    # print(movie_data.head())

    ##########
    ### COUNTRY & ASPECT RATIO
    ##########
    # check number on v
    # print(movie_data["country"].value_counts())

    # most are within top four vals - usa, uk, france, germany. Group the remaining vals into "other" value
    movie_data["country"] = movie_data["country"].where(movie_data["country"].isin(["usa", "uk", "france", "germany"]), "other")

    # print(movie_data["aspect_ratio"].value_counts())

    ##################################################################################################
    ############################################################################### FREQUENCY ENCODING
    ##################################################################################################
    # WHY FREQUENCY ENCODING FOR THE FOLLOWING COLUMNS?
    # we can establish a relationship between # of movies directored and overall success
    # when users input movie data that may include directors, actors, etc that are not in the training data, label encoding would be more difficult to determine what route to take... as opposed to freq encoding, where we can just assign "1" for unknown/new(to training data) directors
    # TODO: sum actors instead of checking each individual column?

    # DIRECTOR_NAME
    director_name_count = movie_data.director_name.value_counts()   # create series that contains frequency of director
    director_name_count = director_name_count.reset_index()         # move index (dir name) as new col
    director_name_count = director_name_count.rename(columns={"index": "director_name", "count": "director_name_count"})   # rename cols for future merge
    director_name_count.loc[director_name_count["director_name"] == "unknown", "director_name_count"] = 0                   # reset the unknown director counts to 0 #TODO: should I do mean instead?
    movie_data = pd.merge(movie_data, director_name_count, left_on="director_name", right_on="director_name")               # merge series with frame (match dirName keys) - only adds count col
    # print(movie_data.head())    # wont delete dirName JUST YET in case its needed
    model.director_name_count = director_name_count

    # ACTORS
    actor1_name_count = movie_data.actor_1_name.value_counts()   # create series that contains frequency of a1
    actor1_name_count = actor1_name_count.reset_index()         # move index (dir name) as new col
    actor1_name_count = actor1_name_count.rename(columns={"index": "actor_1_name", "count": "actor_1_name_count"})   # rename cols for future merge
    actor1_name_count.loc[actor1_name_count["actor_1_name"] == "unknown", "actor_1_name_count"] = 0                   # reset the unknown actor counts to 0 #TODO: should I do mean instead?
    movie_data = pd.merge(movie_data, actor1_name_count, left_on="actor_1_name", right_on="actor_1_name")               # merge series with frame (match dirName keys) - only adds count col
    model.actor1_name_count = actor1_name_count

    actor2_name_count = model.actor2_name_count
    actor2_name_count = movie_data.actor_2_name.value_counts()   # create series that contains frequency of a2
    actor2_name_count = actor2_name_count.reset_index()         # move index (dir name) as new col
    actor2_name_count = actor2_name_count.rename(columns={"index": "actor_2_name", "count": "actor_2_name_count"})   # rename cols for future merge
    actor2_name_count.loc[actor2_name_count["actor_2_name"] == "unknown", "actor_2_name_count"] = 0                   # reset the unknown actor counts to 0 #TODO: should I do mean instead?
    movie_data = pd.merge(movie_data, actor2_name_count, left_on="actor_2_name", right_on="actor_2_name")               # merge series with frame (match dirName keys) - only adds count col
    model.actor2_name_count = actor2_name_count

    actor3_name_count = model.actor3_name_count
    actor3_name_count = movie_data.actor_3_name.value_counts()   # create series that contains frequency of a3
    actor3_name_count = actor3_name_count.reset_index()         # move index (dir name) as new col
    actor3_name_count = actor3_name_count.rename(columns={"index": "actor_3_name", "count": "actor_3_name_count"})   # rename cols for future merge
    actor3_name_count.loc[actor3_name_count["actor_3_name"] == "unknown", "actor_3_name_count"] = 0                   # reset the unknown actor counts to 0 #TODO: should I do mean instead?
    movie_data = pd.merge(movie_data, actor3_name_count, left_on="actor_3_name", right_on="actor_3_name")               # merge series with frame (match dirName keys) - only adds count col
    model.actor3_name_count = actor3_name_count



    # ONE-HOT ENCODING
    categoricals = ["content_rating", "country", "color"]

    for column in categoricals:
        dummies = pd.get_dummies(movie_data[column], prefix=column)
        movie_data = pd.concat([movie_data, dummies], axis=1)
        movie_data.drop(column, axis=1, inplace=True)



    ##################################################################################################
    #################################################################### MODEL ANALYSIS AND COMPARISON
    ##################################################################################################

    backup_movie_data = movie_data.copy() # just in case obj cols are needed for future (user exp)

    ### OBTAIN TRAINING AND TESTING SETS (drop obj cols)
    movie_data = movie_data.drop(columns = ["director_name", "actor_1_name", "actor_2_name", "actor_3_name"], axis=1)
    model.movie_data = movie_data

    # print(movie_data.info())
    x = movie_data.drop("imdb_score", axis=1)
    y = movie_data["imdb_score"]
    # print(movie_data.loc[0])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)


    ### LINEAR REGRESSION MODEL------------------------------------------------------------------------------------
    model_lr = LinearRegression()

    # train
    model_lr.fit(x_train, y_train)

    # predict
    y_pred_lr = model_lr.predict(x_test)

    # evaluate
    # evaluateModel("LINEAR REGRESSION", y_test, y_pred_lr, x_train.shape[0], x_train.shape[1])

    # figures - predicted vs actual
    # createFigures(y_test, y_pred_lr, "Linear Regression", "/linreg_")




    ### RANDOM FOREST MODEL------------------------------------------------------------------------------------
    model_rf = RandomForestRegressor(n_estimators=220) # TRIED 100-300 ... 220 SEEMS TO BE THE BEST TO SETTLE ON
    # model_rf = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    # model_rf = Sequential()
    # model_rf.add(Dense(64, input_dim=x_train.shape[1], activation="relu"))
    #
    # train
    model_rf.fit(x_train, y_train)

    # predict
    y_pred_rf = model_rf.predict(x_test)

    # evaluate
    # printEvaluateModel("RANDOM FOREST", y_test, y_pred_rf, x_train.shape[0], x_train.shape[1])
    rf_stats = get_model_eval(y_test, y_pred_rf, x_train.shape[0], x_train.shape[1])


    model.model = model_rf
    model.model_stats = rf_stats
    model.y_actual = y_test
    model.y_predicted = y_pred_rf

    # figures - predicted vs actual
    # createFigures(y_test, y_pred_rf, "Random Forest", "/rforest_")



    ### DECISION TREE MODEL------------------------------------------------------------------------------------
    # deleted due to RF being drastically better/unnecessary for comparison

    ### GRADIENT BOOST MODEL------------------------------------------------------------------------------------
    # horrible results

    ### NN MODEL------------------------------------------------------------------------------------
    # TODO - try later







    ##################################################################################################
    ########################################################################
    ##################################################################################################

    ##########################
    ########
    ##########################

    # ...................................................................

    ##################################################################################################
    #################### NOTES   NOTES   NOTES   NOTES   NOTES   NOTES   NOTES   NOTES   NOTES   NOTES
    ##################################################################################################

    # ACCESS SERIES INDEX   .............   series.iloc[#] ....... series.loc["label"]


















