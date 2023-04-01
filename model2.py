# Importing the libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load the dataset
df = pd.read_csv(r"C:\Users\bhimsendabby\Downloads\NewIndianFoodDatasetCSV.csv")

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

# f = get_recommendations('Tomato')
# #
# print(f)