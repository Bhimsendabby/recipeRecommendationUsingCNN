"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from PIL import Image, ImageOps
import numpy as np

from PIL import Image
from model import predict_image, category
from tempfile import NamedTemporaryFile
from model2 import get_recommendations


#Set the page configuration for multiple page
st.set_page_config(
    page_title="Main",
    page_icon="👋",
)

st.sidebar.success("Select a Option above.")

#Global variable
listOfRecipes = []

#Headings and title for the page
st.title('Recipe Recommendations')

st.write('Given a list of ingredients, what different recipes can I can make? 🍅')
st.write('For example, say I want to use up some food in my apartment, what can I cook? 🏠 My ML based model will look through over 450 recipes to find matches for you... 🔍 Try it out for yourself below! ⬇️')

st.set_option('deprecation.showfileUploaderEncoding', False)

picture = True
# Take file as input using file_uploader function of streamlit
buffer = st.file_uploader("Upload a JPG File", type=['jpg'])
temp_file = NamedTemporaryFile(delete=False)

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

            st.write(f)
        else:
            st.write("No Vegetable is detected kindly upload a different image")
else:
    st.warning("Kindly upload an Image First", icon="⚠️")







