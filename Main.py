"""
# Main front end file
"""

import pandas as pd
from PIL import Image
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tempfile import NamedTemporaryFile

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from model2 import get_recommendations


#Set the page configuration for multiple page
st.set_page_config(
    page_title="Main",
    page_icon="👋",
)



# vegetable categories or classes on which model has been trained
category = {
    0: 'Bean', 1: 'Bitter_Gourd', 2: 'Bottle_Gourd', 3: 'Brinjal', 4: "Broccoli", 5: 'Cabbage', 6: 'Capsicum',
    7: 'Carrot', 8: 'Cauliflower',
    9: 'Cucumber', 10: 'Papaya', 11: 'Potato', 12: 'Pumpkin', 13: "Radish", 14: "Tomato"
}

def predict_image(filename):
    # Path of the model where model is stored
    path_to_model = r'model_inceptionV3_epoch5.h5'
    print("Loading the model..")

    with st.spinner('Model is being loaded..'):
        # Load model using load_model function of Keras
        model = keras.models.load_model(path_to_model, compile=False)
        model.compile()
        print("Done!")

    #Image loading
    img_ = tf.keras.utils.load_img(filename, target_size=(224, 224))

    #converting image to array
    img_array = tf.keras.utils.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0)
    img_processed /= 255.

    # prediction using already loaded model
    prediction = model.predict(img_processed)

    #finding the maximum value which indicates the highest probablity of the detected object
    index = np.argmax(prediction)

    print(category[index])

    return category[index]




# Load the dataset
df = pd.read_csv(r"D:\NewDownloads\NewIndianFoodDatasetCSV.csv")

# Select the relevant columns
df = df[['TranslatedRecipeName', 'TranslatedIngredients', 'Cuisine', 'Course','TotalTimeInMins','URL']]

# Create a TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform the vectorizer
tfidf_matrix = tfidf.fit_transform(df['TranslatedIngredients'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

nltk.download('stopwords')

def get_recommendations(ingredients, cosine_sim=cosine_sim):
    # Preprocess the ingredients by stemming and removing stop words
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    ingredients = [stemmer.stem(word) for word in ingredients if word not in stop_words]

    # Join the list of ingredients into a string with a space separator
    keywords = ' '.join(ingredients)

    # Create a boolean mask of recipes containing the ingredients in the title or ingredients
    mask = df.apply(lambda x: all(
        keyword in stemmer.stem(x['TranslatedRecipeName'].lower()) or stemmer.stem(keyword) in x[
            'TranslatedIngredients'].lower() for keyword in ingredients), axis=1)

    # Get the indices of the matching recipes
    indices = df[mask].index

    # Get the pairwise similarity scores for all matching recipes
    sim_scores = list(enumerate(cosine_sim[indices].mean(axis=0)))

    # Sort the recipes by similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 10 most similar recipes
    sim_scores = sim_scores[:10]

    # Get the indices of the most similar recipes
    recipe_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar recipes
    return df.iloc[recipe_indices][['TranslatedRecipeName', 'TranslatedIngredients', 'Course', 'TotalTimeInMins', 'URL']]



st.sidebar.success("Select a Option above.")

#Global variable
listOfRecipes = []

#Headings and title for the page
st.title('Recipe Recommendations')

#Text content on the front end app
st.write('Given a list of ingredients, what different recipes can I can make? 🍅')
st.write('For example, say I want to use up some food in my apartment, what can I cook? 🏠 My ML based model will look through over 450 recipes to find matches for you... 🔍 Try it out for yourself below! ⬇️')

#option has been set for the file uploader
st.set_option('deprecation.showfileUploaderEncoding', False)

picture = True
# Take file as input using file_uploader function of streamlit
buffer = st.file_uploader("Upload a JPG File", type=['jpg'])
temp_file = NamedTemporaryFile(delete=False)

#Submit button for the action
result = st.button("Submit")


# Handling the click event of the
if result:
    image = Image.open(buffer)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    if buffer:
        temp_file.write(buffer.getvalue())

        #Calling the predict function from our model
        veg = predict_image(temp_file.name)
        st.write("")
        st.write("Classifying...")
        if veg in list(category.values()):
            st.text("Recommended Recipes Are....")
            st.write(veg)

            #Calling the recommendation function from recommendation model
            f = get_recommendations(veg)

            # style for the table
            # th_props = [
            #     ('font-size', '14px'),
            #     ('text-align', 'center'),
            #     ('font-weight', 'bold'),
            #     ('color', '#6d6d6d'),
            #     ('background-color', '#f7ffff')
            # ]
            #
            # # font size for the table text
            # td_props = [
            #     ('font-size', '18px')
            # ]
            #
            # # dict object for styling
            # styles = [
            #     dict(selector="th", props=th_props),
            #     dict(selector="td", props=td_props),
            # ]
            #
            # # table
            # f = f.style.set_properties(**{'text-align': 'left'}).set_table_styles(styles)
            #
            # #show recipes
            # st.table(f)
            st.write(f)
        else:
            st.write("No Vegetable is detected kindly upload a different image")
else:
    # warning when no image found
    st.warning("Kindly upload an Image First", icon="⚠️")







