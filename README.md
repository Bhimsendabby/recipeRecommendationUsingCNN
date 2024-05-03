# Culinary Recommendation System: Cooking Made Easy for International Students

## Introduction:
Our project addresses the culinary dilemmas faced by international students, who often find themselves in unfamiliar kitchens with limited knowledge of recipes and ingredients. Through the innovative use of Convolutional Neural Networks (CNNs), our system offers a seamless solution for meal planning and recipe recommendations. By simply uploading images of available vegetables or manually inputting ingredients, users can access a curated list of top recipes tailored to their resources and preferences.

## Purpose:
The primary motivation behind our project is to empower international students with the confidence and capability to cook nutritious and delicious meals, even with minimal culinary experience or ingredients at hand. Recognizing the challenges of adapting to new environments, especially in terms of food culture, our system serves as a digital sous chef, guiding users through the cooking process with ease and efficiency.

## Key Features:
1. **Image Recognition:** Leveraging CNN technology, our system accurately identifies vegetables from uploaded images, eliminating the need for manual ingredient lists.
2. **Ingredient-Based Recommendations:** Whether users have a fully stocked pantry or just a few items in the fridge, our system generates recipe suggestions based on available ingredients, maximizing resource utilization and minimizing food waste.
3. **User-Friendly Interface:** With an intuitive and user-friendly interface, our project caters to individuals of all skill levels, from novice cooks to seasoned chefs, providing step-by-step instructions and visual aids for each recipe.

## Conclusion:
In conclusion, our Culinary Recommendation System revolutionizes the way international students approach cooking, offering a practical solution for meal planning and recipe discovery. By harnessing the power of CNNs and machine learning algorithms, we aim to foster culinary independence and creativity among users, enabling them to explore new flavors and cuisines with confidence and convenience.

##Upload Image
![image](https://github.com/Bhimsendabby/recipeRecommendationUsingCNN/assets/35491121/3f044cd3-fe5c-4d0b-a948-0699877e7b4c)

##Vegetable Detected
![image](https://github.com/Bhimsendabby/recipeRecommendationUsingCNN/assets/35491121/59e30395-6e4a-47f4-a7cf-1272fa08f749)

##Recipe Recommendation
![image](https://github.com/Bhimsendabby/recipeRecommendationUsingCNN/assets/35491121/07a861f7-6425-4049-8d6e-a14da7f3ec66)



## Libraries Used:

- **TensorFlow:** TensorFlow is an open-source machine learning framework developed by Google. We utilized TensorFlow for implementing machine learning models, including Convolutional Neural Networks (CNNs), to handle tasks such as image recognition within our project.

- **OpenCV:** OpenCV (Open Source Computer Vision Library) is a widely-used computer vision library with extensive capabilities for image processing and analysis. We integrated OpenCV into our project to perform various image-related tasks, such as preprocessing and manipulation.

- **Pandas:** Pandas is a powerful data manipulation and analysis library for Python. We employed Pandas to manage and manipulate tabular data efficiently, facilitating tasks such as data cleaning, transformation, and analysis within our project.

- **NumPy:** NumPy is a fundamental package for scientific computing in Python, providing support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays. We utilized NumPy for numerical computations and array manipulation tasks within our project.

- **Tempfile:** Tempfile is a Python library for generating temporary files and directories. We used Tempfile to handle temporary file operations within our project, such as saving and managing intermediate files during image processing tasks.

- **Streamlit:** Streamlit is an open-source Python library used for building interactive web applications for data science and machine learning projects. We leveraged Streamlit to develop the user interface (UI) of our project, enabling users to interact with the system easily through a web browser.

- **Pillow:** Pillow is a Python Imaging Library (PIL) fork, providing support for opening, manipulating, and saving many different image file formats. We incorporated Pillow into our project to handle image loading, processing, and display tasks, enhancing the overall functionality of the image-related components.

To run our project on your local machine, follow these steps:

1. Ensure you have Python installed on your machine. If not, download and install Python from [python.org](https://www.python.org/).

2. Clone or download the project repository to your local machine.

3. Navigate to the project directory in your terminal.

4. Install the required Python libraries listed in the `requirements.txt` file. You can do this by running the following command in your terminal:

pip install -r requirements.txt

5. Once all dependencies are installed, you can start the Streamlit application by running the following command in your terminal:

streamlit run Main.py


6. This command will start a local server and open the project in your default web browser. You can now interact with the project interface to upload images or input ingredients manually and receive recipe recommendations based on your input.

Note: Make sure you have a stable internet connection during the installation process to download any required packages.




