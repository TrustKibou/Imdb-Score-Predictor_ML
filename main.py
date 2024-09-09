import customtkinter as ctk
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from model_prep import Model, create_figures, preprocessAndTrain
import random

# APPLICATION STYLING
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

window = ctk.CTk()    # primary window!
window.title("IMDb Score Predictor")
window.geometry("500x400")

# FRAME INITIALIZATION
loading_frame = ctk.CTkFrame(window, width=500, height=400)
home_frame = ctk.CTkFrame(window, width=500, height=400)
predict_frame = ctk.CTkFrame(window, width=500, height=400)
stats_frame = ctk.CTkFrame(window, width=500, height=400)
plots_frame = ctk.CTkFrame(window, width=500, height=400)


##############################################################################################################################################
############################################################################################################### LOADING FRAME & MODEL TRAINING
##############################################################################################################################################


###################################### SWITCH VIEWED FRAME
def raise_frame(display_frame, size):

    # HIDE ALL OTHER MAIN FRAMES
    for frame in (loading_frame, home_frame, predict_frame, stats_frame, plots_frame):
        frame.pack_forget()

    # SHOW CURRENT FRAME
    display_frame.pack(fill="both", expand=True)

    # ADJUST WINDOW SIZE
    window.geometry(size)



###################################### TRAIN MODEL
def get_model():
    global model

    model = Model()
    preprocessAndTrain(model)

    update_stats_frame(model)
    update_plots_frame(model)


    raise_frame(home_frame, "500x400")


###################################### FIRST ---> SHOW LOADING FRAME, THEN TRAIN MODEL, THEN SHOW HOME FRAME
load_label = ctk.CTkLabel(loading_frame, text="Training model... please wait", font=("Tekton Pro", 16, "bold"))
load_label.pack(pady=30)

# CENTER APP TODO: check if this works properly for teammates (goal: centered in launching display)
x = (window.winfo_screenwidth() // 2)
y = (window.winfo_screenheight() // 2) - 200
loading_frame.pack(fill="both", expand=True)
window.geometry(f'+{x}+{y}')
# raise_frame(loading_frame, "500x400")

window.after(2000, get_model)

# window.after(1000, train_model)







##############################################################################################################################################
#################################################################################################################################### FUNCTIONS
##############################################################################################################################################


###################################### PREDICT SCORE
def predict_score(inputs):
    data = {field: ent.get().lower() if isinstance(ent.get(), str) else ent.get() for field, ent in inputs.items()}

    # QUICK ACCESS VARS
    movie_data = model.movie_data

    # CREATE NEW DATA FRAME FOR USER'S MOVIE INFO
    user_data = pd.DataFrame(columns = movie_data.columns).astype(movie_data.dtypes)

    # POPULATE ENTIRE ROW WITH ZEROES/FALSE (DONE THIS WAY TO >>PRESERVE ORIGINAL COL DATA TYPES<<, WHICH WAS A HUGE HUGE HUGE HASSLE)
    defaults = {}

    for col, dtype in movie_data.dtypes.items():
        if dtype.kind in ["i", "f"]:           # INTS, FLOATS
            defaults[col] = 0
        elif dtype.kind == "b":                # BOOLEAN
            defaults[col] = False
        else:
            defaults[col] = None         # JIC

    # Append the row with default values to the DataFrame
    user_data = user_data._append(defaults, ignore_index=True)      # not suggested... don't care, may fix later

    # SIMPLE VARS (less d cleaning)
    user_data.loc[0, "title_year"] = int(data["Release Year"]) if data["Release Year"] else 2024
    user_data.loc[0, "duration"] = int(data["Duration (in minutes)"]) if data["Duration (in minutes)"] else movie_data["duration"].mean().astype(int)
    user_data.loc[0, "aspect_ratio"] = float(data["Aspect Ratio"]) if data["Aspect Ratio"] else movie_data.aspect_ratio.mode().iloc[0]
    user_data.loc[0, "budget"] = int(data["Budget"]) if data["Budget"] else movie_data["budget"].mean().astype(int)
    user_data.loc[0, "gross"] = int(data["Gross"]) if data["Gross"] else movie_data["gross"].mean().astype(int)
    user_data.loc[0, "director_facebook_likes"] = int(data["Director's Total FB or IG Likes"]) if data["Director's Total FB or IG Likes"] else movie_data["director_facebook_likes"].mean().astype(int)
    user_data.loc[0, "movie_facebook_likes"] = int(data["Movie's Total FB or IG Likes"]) if data["Movie's Total FB or IG Likes"] else movie_data["movie_facebook_likes"].mean().astype(int)
    user_data.loc[0, "actor_1_facebook_likes"] = int(data["Actor 1 Total FB or IG Likes"]) if data["Actor 1 Total FB or IG Likes"] else movie_data["actor_1_facebook_likes"].mean().astype(int)
    user_data.loc[0, "actor_2_facebook_likes"] = int(data["Actor 2 Total FB or IG Likes"]) if data["Actor 2 Total FB or IG Likes"] else movie_data["actor_2_facebook_likes"].mean().astype(int)
    user_data.loc[0, "actor_3_facebook_likes"] = int(data["Actor 3 Total FB or IG Likes"]) if data["Actor 3 Total FB or IG Likes"] else movie_data["actor_3_facebook_likes"].mean().astype(int)
    user_data.loc[0, "cast_total_facebook_likes"] = user_data.loc[0, "actor_1_facebook_likes"] + user_data.loc[0, "actor_2_facebook_likes"] + user_data.loc[0, "actor_3_facebook_likes"]
    user_data.loc[0, "facenumber_in_poster"] = int(data["Number of Faces in Movie Poster"]) if data["Number of Faces in Movie Poster"] else movie_data["facenumber_in_poster"].mean().astype(int)
    user_data.loc[0, "num_imdb_voted_users"] = int(data["Number of IMDb Voted Users"]) if data["Number of IMDb Voted Users"] else movie_data["num_imdb_voted_users"].mean().astype(int)
    user_data.loc[0, "num_critic_for_reviews"] = int(data["Number of Critic Reviews"]) if data["Number of Critic Reviews"] else movie_data["num_critic_for_reviews"].mean().astype(int)
    user_data.loc[0, "num_user_for_reviews"] = int(data["Number of User Reviews"]) if data["Number of User Reviews"] else movie_data["num_user_for_reviews"].mean().astype(int)
    user_data.loc[0, "profit"] = user_data.loc[0, "gross"] - user_data.loc[0, "budget"]

    # COMPLEX VARS - FREQUENCY (func for code reuse for dir/act names) (not the most readable though)
    def check_count(users_name, col_name, dframe, dframe_name):
        if users_name in dframe[col_name].values:
            return dframe.loc[dframe[col_name] == users_name, dframe_name].iloc[0]
        else:
            return 0

    # COMPLEX VARS - FREQUENCY
    user_data.loc[0, "director_name_count"] = check_count(data["Director's Name"], "director_name", model.director_name_count, "director_name_count")
    user_data.loc[0, "actor_1_name_count"] = check_count(data["Actor 1 Name"], "actor_1_name", model.actor1_name_count, "actor_1_name_count")
    user_data.loc[0, "actor_2_name_count"] = check_count(data["Actor 2 Name"], "actor_2_name", model.actor2_name_count, "actor_2_name_count")
    user_data.loc[0, "actor_3_name_count"] = check_count(data["Actor 3 Name"], "actor_3_name", model.actor3_name_count, "actor_3_name_count")

    # COMPLEX VARS - GENRE
    genre_list = ["genre_" + data["Genre1"], "genre_" + data["Genre2"]]

    if genre_list[0] in user_data.columns:
        user_data.loc[0, genre_list[0]] = 1
    if genre_list[1] in user_data.columns:
        user_data.loc[0, genre_list[1]] = 1

    # COMPLEX VARS - CONTENT RATING
    user_data.loc[0, ("content_rating_" + data["Content Rating"])] = True

    # COMPLEX VARS - COUNTRY
    user_data.loc[0, ("country_" + data["Country"])] = True

    # COMPLEX VARS - COLOR
    user_data.loc[0, ("color_" + data["Color"])] = True

    ############# NOW PREDICT!
    user_data.drop(columns = "imdb_score", axis = 1, inplace = True)
    prediction = model.model.predict(user_data)
    display_prediction(prediction[0])
    # print("Predicted IMDb Score:", prediction[0])


###################################### DISPLAY PREDICTION
def display_prediction(score):
    predict_window = ctk.CTkToplevel()
    predict_window.title("")
    predict_window.attributes("-topmost", True)     # ensure new window stays on top of old window
    predict_window.focus_force()                    # extra assurances :D

    # CENTER WINDOW
    x = window.winfo_x() + window.winfo_width() // 2 - 150
    y = window.winfo_y() + window.winfo_height() // 2 - 165
    predict_window.geometry(f"200x230+{x}+{y}")

    # TITLE LABEL
    plabel = ctk.CTkLabel(predict_window, text="The predicted score is...", font=("Tekton Pro", 10))
    plabel.pack(pady=10, padx=10)

    # SEPARATOR #1
    s1 = ctk.CTkFrame(predict_window, height=2, bg_color="white")
    s1.pack(fill="x", padx=15, pady=10)

    # SCORE
    score = format(score, ".2f")
    plabel = ctk.CTkLabel(predict_window, text=str(score), font=("Tekton Pro", 26, "bold"), text_color="#911d29")
    plabel.pack(pady=20, padx=30)

    # SEPARATOR #2
    s2 = ctk.CTkFrame(predict_window, height=2, bg_color="white")
    s2.pack(fill="x", padx=10, pady=10)

    # CLOSE BUTTON
    button = ctk.CTkButton(predict_window, text="Close Prediction", command=predict_window.destroy)
    button.pack(pady=15)


###################################### VALIDATION - INT & FLOAT - stackoverflow ty
def check_integer(P):
    if P.isdigit() or P == "":
        return True

    return False

def check_float(P):
    if P.strip() == "":
        return True

    try:
        float(P)
        return True
    except ValueError:
        return False

###################################### VALIDATION - PURELY FOR PLACEHOLDER
def enable_validation(event):
    ent = event.widget
    if ent.get() == "":
        ent.configure(validate = "key")

def disable_validation(event):
    ent = event.widget
    ent.configure(validate = "none")



##############################################################################################################################################
################################################################################################################################### HOME FRAME
##############################################################################################################################################

# TITLE
welcome_label = ctk.CTkLabel(home_frame, text="Welcome to the IMDb Score Predictor!", font=("Tekton Pro", 16, "bold"))
welcome_label.pack(pady=20)

# SEPARATOR (TITLE FROM CONTENT)
separator = ctk.CTkFrame(home_frame, height=2, bg_color="white")
separator.pack(fill="x", padx=20, pady=10)

# SUB-TITLE
choice_label = ctk.CTkLabel(home_frame, text="What would you like to do?")
choice_label.pack(pady=10)

# 4 BUTTONS - PREDICT
button_predict = ctk.CTkButton(home_frame, text="Predict a Score", command=lambda: raise_frame(predict_frame, "550x700"))
button_predict.pack(fill="x", padx=80, pady=5)

# 4 BUTTONS - SHOW STATS
button_stats = ctk.CTkButton(home_frame, text="Show Model Stats", command=lambda: raise_frame(stats_frame, "500x400"))
button_stats.pack(fill="x", padx=80, pady=5)

# 4 BUTTONS - VIEW PLOTS
button_plots = ctk.CTkButton(home_frame, text="View Plots", command=lambda: raise_frame(plots_frame, "550x400"))
button_plots.pack(fill="x", padx=80, pady=5)

# 4 BUTTONS - EXIT
button_exit = ctk.CTkButton(home_frame, text="Exit", command=window.destroy, fg_color="#6b0713", hover_color="#9e1122")
button_exit.pack(fill="x", padx=80, pady=50)

##############################################################################################################################################
########################################################################################################################## PREDICTION FRAME
##############################################################################################################################################

########## CONFIGURE CANVAS FOR SCROLLBAR

# CREATE CANVAS WITHIN FRAME - PACK
canvas = ctk.CTkCanvas(predict_frame, bg="#2b2b2b", highlightthickness=0, relief="ridge")
canvas.pack(side="left", fill="both", expand=True)

# ADD SCROLLBAR
scrollbar = ctk.CTkScrollbar(predict_frame, command=canvas.yview)
scrollbar.pack(side="right", fill="y")

# CONFIGURE SCROLLBAR
canvas.configure(yscrollcommand=scrollbar.set)
canvas.yview_moveto(0)      # FIX SCROLLBAR ISSUE!!! KEEPS IT AT TOP
canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

# CREATE SUB-FRAME FOR FUTURE LABELS/INPUT
predict_scroll_frame = ctk.CTkFrame(canvas)
canvas.create_window((0, 0), window=predict_scroll_frame, anchor="nw")

#########################################################################################################################################
########################################################################################################################### PREDICT FRAME
#########################################################################################################################################

def get_placeholder(label):

    # The way I have this set up is quite messy at the moment... if I change a label, there are SEVERAL places I have to change it. TODO may fix this later, though it'd be a lot of extra work
    ph_list = {
        "Movie Title": "My Movie Title",
        "Release Year": "2024 (integers only)",
        "Duration (in minutes)": "124 (integers only)",
        "Aspect Ratio": "2.4 (floats only)",
        "Budget": "350000 (integers only)",
        "Gross": "9000000 (integers only)",
        "Director's Name": "FirstName LastName",
        "Actor 1 Name": "FirstName LastName",
        "Actor 2 Name": "FirstName LastName",
        "Actor 3 Name": "FirstName LastName",
        "Movie's Total FB or IG Likes": "500 (integers only)",
        "Director's Total FB or IG Likes": "500 (integers only)",
        "Actor 1 Total FB or IG Likes": "500 (integers only)",
        "Actor 2 Total FB or IG Likes": "500 (integers only)",
        "Actor 3 Total FB or IG Likes": "500 (integers only)",
        "Number of Faces in Movie Poster": "4 (integers only)",
        "Number of IMDb Voted Users": "300 (integers only)",
        "Number of Critic Reviews": "10 (integers only)",
        "Number of User Reviews": "100 (integers only)"
    }

    return ph_list[label]

def generate_random_movie(fields, movies):

    # GET RANDOM MOVIE
    r_movie = movies.iloc[random.randint(0, len(movies)-1)]

    # CLEAR ALL FIELDS
    reset_prediction(fields)    # forgot I already had a clear funct;

    # POPULATE FIELDS - deleted my brute force method... going to try MAPPING!
    # gui2csv = {
    #     "Movie Title": "movie_title",
    #     "Release Year": "title_year",
    #     "Duration (in minutes)": "duration",
    #     "Aspect Ratio": "aspect_ratio",
    #     "Budget": "budget",
    #     "Gross": "gross",
    #     "Director's Name": "director_name",
    #     "Actor 1 Name": "actor_1_name",
    #     "Actor 2 Name": "actor_2_name",
    #     "Actor 3 Name": "actor_3_name",
    #     "Movie's Total FB or IG Likes": "movie_facebook_likes",
    #     "Director's Total FB or IG Likes": "director_facebook_likes",
    #     "Actor 1 Total FB or IG Likes": "actor_1_facebook_likes",
    #     "Actor 2 Total FB or IG Likes": "actor_2_facebook_likes",
    #     "Actor 3 Total FB or IG Likes": "actor_3_facebook_likes",
    #     "Number of Faces in Movie Poster": "facenumber_in_poster",
    #     "Number of IMDb Voted Users": "num_imdb_voted_users",
    #     "Number of Critic Reviews": "num_critic_for_reviews",
    #     "Number of User Reviews": "num_user_for_reviews"
    # }


    # ENTRIES-------------
    fields["Movie Title"].insert(0, r_movie["movie_title"])
    fields["Release Year"].insert(0, r_movie["title_year"])
    fields["Duration (in minutes)"].insert(0, r_movie["duration"])
    fields["Aspect Ratio"].insert(0, r_movie["aspect_ratio"])
    fields["Budget"].insert(0, r_movie["budget"])
    fields["Gross"].insert(0, r_movie["gross"])
    fields["Director's Name"].insert(0, r_movie["director_name"])
    fields["Actor 1 Name"].insert(0, r_movie["actor_1_name"])
    fields["Actor 2 Name"].insert(0, r_movie["actor_2_name"])
    fields["Actor 3 Name"].insert(0, r_movie["actor_3_name"])
    fields["Movie's Total FB or IG Likes"].insert(0, r_movie["movie_facebook_likes"])
    fields["Director's Total FB or IG Likes"].insert(0, r_movie["director_facebook_likes"])
    fields["Actor 1 Total FB or IG Likes"].insert(0, r_movie["actor_1_facebook_likes"])
    fields["Actor 2 Total FB or IG Likes"].insert(0, r_movie["actor_2_facebook_likes"])
    fields["Actor 3 Total FB or IG Likes"].insert(0, r_movie["actor_3_facebook_likes"])
    fields["Number of Faces in Movie Poster"].insert(0, r_movie["facenumber_in_poster"])
    fields["Number of IMDb Voted Users"].insert(0, r_movie["num_imdb_voted_users"])
    fields["Number of Critic Reviews"].insert(0, r_movie["num_critic_for_reviews"])
    fields["Number of User Reviews"].insert(0, r_movie["num_user_for_reviews"])

    ### DROPDOWNS--------
    # GENRES
    genres = r_movie["genres"].split("|")
    g_num = 0
    fields["Genre1"].set("Other")      # just in case no genre found (for both cols)
    fields["Genre2"].set("Other")      # just in case no genre found (for both cols)

    #... CHANGE GENRE OPT
    for genre in genres:
        if g_num == 2:
            break

        genre = genre.strip()

        if genre in genre_options:
            fields["Genre" + str(g_num+1)].set(genre)
            g_num += 1

        if g_num == 2:
            break



    # CONTENT RATING
    fields["Content Rating"].set(r_movie["content_rating"].strip())

    # COUNTRY
    country = r_movie["country"].strip()

    if country in country_options:
        fields["Country"].set(country)
    else:
        fields["Country"].set("Other")

    # COLOR
    fields["Color"].set(r_movie["color"].strip())




# TODO - put predict stuff into function? I ended up having to do that for the other frames due to implementing the loading_frame later, soo.. to be consistent?
# TITLE
label = ctk.CTkLabel(predict_scroll_frame, text="Please enter as much of the following information as you can: ", font=("Tekton Pro", 16, "bold"), padx=20, pady=20)
label.grid(row=0, column=0, columnspan=3, sticky="e", padx=5, pady=2)

# READ CSV FOR RANDOMIZER
rmovies_csv = pd.read_csv("datasets/movies_testing.csv")

# INDIVIDUAL LABELS
input_labels = [
    "Movie Title",
    "Release Year",
    "Duration (in minutes)",
    "Genre1",
    "Genre2",
    "Aspect Ratio",
    "Budget",
    "Gross",
    "Content Rating",
    "Country",
    "Color",
    "Director's Name",
    "Actor 1 Name",
    "Actor 2 Name",
    "Actor 3 Name",
    "Movie's Total FB or IG Likes",
    "Director's Total FB or IG Likes",
    "Actor 1 Total FB or IG Likes",
    "Actor 2 Total FB or IG Likes",
    "Actor 3 Total FB or IG Likes",
    "Number of Faces in Movie Poster",
    "Number of IMDb Voted Users",
    "Number of Critic Reviews",
    "Number of User Reviews"
    ]

integer_labels = ["Release Year", "Duration (in minutes)", "Budget", "Gross", "Movie's Total FB or IG Likes", "Director's Total FB or IG Likes", "Actor 1 Total FB or IG Likes", "Actor 2 Total FB or IG Likes", "Actor 3 Total FB or IG Likes", "Number of Faces in Movie Poster", "Number of IMDb Voted Users", "Number of Critic Reviews", "Number of User Reviews"]
float_labels = ["Aspect Ratio"]

genre_options = ["Action", "Adventure", "Animation", "Biography", "Comedy", "Crime", "Documentary", "Drama", "Family", "Fantasy", "Film-Noir", "History", "Horror", "Music", "Musical", "Mystery", "Other", "Romance", "Sci-Fi", "Short", "Sport", "Thriller", "War", "Western"]
country_options = ["France", "Germany", "Other", "UK", "USA"]
content_rating_options = ["G", "PG", "PG-13", "R", "NC-17", "Unrated"]

user_movie_data = {}

validate_int = window.register(check_integer), "%P"
validate_float = window.register(check_float), "%P"

# CREATE LABELS & INPUT FIELDS
for i, label_text in enumerate(input_labels, start=1):

    # LABEL TEXT
    label = ctk.CTkLabel(predict_scroll_frame, text=label_text)
    label.grid(row=i, column=0, sticky="e", padx=5, pady=2)

    # INPUT FIELD - COLOR
    if label_text == "Color":

        color_choice = ctk.StringVar(value="Color")
        entry = ctk.CTkOptionMenu(predict_scroll_frame, variable=color_choice, values=["Color", "Black and White"], width=200, fg_color="#5a7e9e")
        entry.grid(row=i, column=1, sticky="w", padx=5, pady=2)

    # INPUT FIELD - GENRE1
    elif label_text == "Genre1":
        g1_choice = ctk.StringVar(value="Other")
        entry = ctk.CTkOptionMenu(predict_scroll_frame, variable=g1_choice, values=genre_options, width=200, fg_color="#5a7e9e")
        entry.grid(row=i, column=1, sticky="w", padx=5, pady=2)

    # INPUT FIELD - GENRE2
    elif label_text == "Genre2":
        g2_choice = ctk.StringVar(value="Other")
        entry = ctk.CTkOptionMenu(predict_scroll_frame, variable=g2_choice, values=genre_options, width=200, fg_color="#5a7e9e")
        entry.grid(row=i, column=1, sticky="w", padx=5, pady=2)

    # INPUT FIELD - COUNTRY
    elif label_text == "Country":
        country_choice = ctk.StringVar(value="USA")
        entry = ctk.CTkOptionMenu(predict_scroll_frame, variable=country_choice, values=country_options, width=200, fg_color="#5a7e9e")
        entry.grid(row=i, column=1, sticky="w", padx=5, pady=2)

    # INPUT FIELD - CONTENT RATING
    elif label_text == "Content Rating":
        content_rating_choice = ctk.StringVar(value="PG-13")
        entry = ctk.CTkOptionMenu(predict_scroll_frame, variable=content_rating_choice, values=content_rating_options, width=200, fg_color="#5a7e9e")
        entry.grid(row=i, column=1, sticky="w", padx=5, pady=2)

    # ALL OTHER INPUT FIELDS
    else:
        # INPUT FIELD - VALIDATE PROPER ENTRY
        if label_text in integer_labels:
            entry = ctk.CTkEntry(predict_scroll_frame, width=200, placeholder_text=get_placeholder(label_text), validatecommand=validate_int)
            entry.bind("<Key>", enable_validation)
            entry.bind("<FocusOut>", disable_validation)
        elif label_text in float_labels:
            entry = ctk.CTkEntry(predict_scroll_frame, width=200, placeholder_text=get_placeholder(label_text), validatecommand=validate_float)
            entry.bind("<Key>", enable_validation)
            entry.bind("<FocusOut>", disable_validation)
        else:
            entry = ctk.CTkEntry(predict_scroll_frame, width=200, placeholder_text=get_placeholder(label_text))

        # PLACE ON GRID
        entry.grid(row=i, column=1, sticky="w", padx=5, pady=2)

        # SAVE FOR PREDICTION
    user_movie_data[label_text] = entry

# FOR BUTTONS - LENGTH
length = len(user_movie_data)

# BUTTON - PREDICT MY MOVIE
button_predict = ctk.CTkButton(predict_scroll_frame, text="Predict", command=lambda: predict_score(user_movie_data))
button_predict.grid(row=length + 1, column=1, columnspan=1, pady=10)

# BUTTON - RETURN TO HOME PAGE
button_home = ctk.CTkButton(predict_scroll_frame, text="Return Home", command=lambda: raise_frame(home_frame, "500x400"), width=100, height=10)
button_home.grid(row=length + 3, column=0, columnspan=1, pady=10, sticky="E")

def reset_prediction(fields):
    for lbl, field in fields.items():
        if isinstance(field, ctk.CTkEntry):
            field.delete(0, "end")    # does actual deletion of text data in fields
            field._activate_placeholder()               # not suggested, but done by developer as a temporary solution sooo...
            entry.master.focus()                        # brings focus back to reset button, to my understanding?


# BUTTON - GENERATE RANDOM MOVIE (from provided file)
button_reset = ctk.CTkButton(predict_scroll_frame, text="Generate Random", command=lambda: generate_random_movie(user_movie_data, rmovies_csv))
button_reset.grid(row=length + 2, column=1, columnspan=1, pady=10)

# BUTTON - RESET
button_reset = ctk.CTkButton(predict_scroll_frame, text="Reset", command=lambda: reset_prediction(user_movie_data))
button_reset.grid(row= length + 3, column=1, columnspan=1, pady=10)




#########################################################################################################################################
############################################################################################################################## STATISTICS
#########################################################################################################################################

def update_stats_frame(model):

    model_stats = model.model_stats

    # TITLE LABE
    label_stats_title = ctk.CTkLabel(stats_frame, text="Random Forest Model Stats", font=("Tekton Pro", 16, "bold"))
    label_stats_title.pack(pady=20)

    # STATS-------------
    label_mse = ctk.CTkLabel(stats_frame, text="MSE is " + str(model_stats["mse"]), font=("Tekton Pro", 14))
    label_mse.pack(pady=5)

    label_rmse = ctk.CTkLabel(stats_frame, text="RMSE is " + str(model_stats["rmse"]), font=("Tekton Pro", 14))
    label_rmse.pack(pady=5)

    label_mae = ctk.CTkLabel(stats_frame, text="MAE is " + str(model_stats["mae"]), font=("Tekton Pro", 14))
    label_mae.pack(pady=5)

    label_mape = ctk.CTkLabel(stats_frame, text="MAPE is " + str(model_stats["mape"]), font=("Tekton Pro", 14))
    label_mape.pack(pady=5)

    label_r2 = ctk.CTkLabel(stats_frame, text="R^2 is " + str(model_stats["r2"]), font=("Tekton Pro", 14))
    label_r2.pack(pady=5)

    label_r2adj = ctk.CTkLabel(stats_frame, text="R^2 Adjusted is " + str(model_stats["r2adj"]), font=("Tekton Pro", 14))
    label_r2adj.pack(pady=5)

    # BUTTON - RETURN TO HOME PAGE (ugly name because of predicted not being moduled)
    b_home = ctk.CTkButton(stats_frame, text="Return Home", command=lambda: raise_frame(home_frame, "500x400"), width=100, height=10)
    b_home.pack(pady=15)



#########################################################################################################################################
############################################################################################################################# PLOTS FRAME
#########################################################################################################################################

def update_plots_frame(mdl):

    # DECIDE WHICH FIGURE NEEDS TO BE SHOWN AND HIGHLIGHT STATUS (USABILITY) OF NEXT/PREV BUTTONS
    def show_figure(index):

        # HIDE ALL FIGS
        for fig in canvas_list:
            fig.pack_forget()

        # SHOW CORRECT FIG
        canvas_list[index].pack(side="top", fill="x", expand=False) # expansion prob

        # ADJUST BUTTONS BASED ON WHICH FIG IS SHOWING - WILL ADAPT IF MORE PLOTS ARE CREATED LATER!!
        b_prev.configure(state="normal" if index > 0 else "disabled")                       # only activate prev for 1-(n-1)
        b_next.configure(state="normal" if index < len(canvas_list) - 1 else "disabled")    # only activate next for 0-(n-2)



    # BUTTON FUNCS --- NAVIGATE THROUGH FIGURES AND ADJUST BUTTON STATES
    def navigate(step):
        current_index[0] += step
        show_figure(current_index[0])



    # CREATE GRAPHS
    figures_list = create_figures(mdl.y_actual, mdl.y_predicted, "Random Forest")

    # SAFE DESTR0Y - (fix graphs appearing ontop)
    for widg in plots_frame.winfo_children():
        widg.destroy()

    # BUTTONS FRAME (BOTTOM)
    navbuttons_frame = ctk.CTkFrame(plots_frame)
    navbuttons_frame.pack(side="bottom", fill="x")

    # VARS
    canvas_list = []        # will hold plots
    current_index = [0]     # for fig/btns

    # ADD FIGURES TO FRAMES
    for figure in figures_list:
        canv = FigureCanvasTkAgg(figure, master=plots_frame)
        canv.draw() # create

        widg = canv.get_tk_widget()
        # widg.pack(side="top", fill="both", expand=True)   # expansion prob
        canvas_list.append(widg)

    # NAVIGATION BUTTON FRAME DETAILS
    # --> SET COLUMN TOTAL (for display purposes)
    navbuttons_frame.grid_columnconfigure(0, weight=1)
    navbuttons_frame.grid_columnconfigure(1, weight=1)
    navbuttons_frame.grid_columnconfigure(2, weight=1)

    # ---> HOME
    b_home = ctk.CTkButton(navbuttons_frame, text="Return Home", command=lambda: raise_frame(home_frame, "500x400"), width=100, height=10)
    b_home.grid(row=0, column=0, sticky="w", padx=10, pady=10)

    # ---> PREVIOUS (start disabled)
    b_prev = ctk.CTkButton(navbuttons_frame, text="Previous", state="disabled", command=lambda: navigate(-1))
    b_prev.grid(row=0, column=1, sticky="e", padx=10, pady=10)

    # ---> NEXT
    b_next = ctk.CTkButton(navbuttons_frame, text="Next", command=lambda: navigate(1))
    b_next.grid(row=0, column=2, sticky="E", padx=10, pady=10)

    # START WITH FIRST FIGURE (ACTvPRED)
    show_figure(0)


#########################################################################################################################################
############################################################################################################################### EXTERNALS
#########################################################################################################################################

# FRAME TO DISPLAY FIRST
# raise_frame(home_frame, "500x400")

# KEEP PROGRAM IN GUI
window.mainloop()


