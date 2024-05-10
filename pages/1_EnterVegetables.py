import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load the dataset
df = pd.read_csv(r"NewIndianFoodDatasetCSV.csv")

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

#Set the page configuration for multiple page
st.set_page_config(page_title="User Input Recommendation", page_icon="📈")

#Markdown to print the required content
st.markdown("# Enter the Vegetables Names To Get Recommendation Of Recipes")
st.sidebar.header("Enter Vegetables Manually")
st.write("""This demo illustrates where a user can enter the vegetables names in the input box""")

# For taking the text input for the recommendation instead of image
vegs = st.text_input("Enter some text",placeholder="Enter the Vegetables names by comma separated values")

#split the vegetables
st.write("Enter",vegs)

st.markdown(""" <style> .font {
        font-size:100px;} 
        </style> """, unsafe_allow_html=True)

if vegs:
    splitedVeg = vegs.split(",")
    #To submit the entered vegetables
    textResult = st.button('Recommend')
    st.text("Recommended Recipes Are....")
    if textResult:
        st.write("")
        st.write("Classifying...")
        f = get_recommendations(splitedVeg)
        # style for the table
        th_props = [
            ('font-size', '14px'),
            ('text-align', 'center'),
            ('font-weight', 'bold'),
            ('color', '#6d6d6d'),
            ('background-color', '#f7ffff')
        ]

        #font size for the table text
        td_props = [
            ('font-size', '18px')
        ]

        #dict object for styling
        styles = [
            dict(selector="th", props=th_props),
            dict(selector="td", props=td_props),
        ]

        # table
        f = f.style.set_properties(**{'text-align': 'left'}).set_table_styles(styles)
        st.table(f)
