import numpy as np
import tensorflow as tf
from tensorflow import keras


# vegetable categories or classes on which model has been trained
category = {
    0: 'Bean', 1: 'Bitter_Gourd', 2: 'Bottle_Gourd', 3: 'Brinjal', 4: "Broccoli", 5: 'Cabbage', 6: 'Capsicum',
    7: 'Carrot', 8: 'Cauliflower',
    9: 'Cucumber', 10: 'Papaya', 11: 'Potato', 12: 'Pumpkin', 13: "Radish", 14: "Tomato"
}

# Prediction function which take image as input and return string as output which is the vegetable name

def predict_image(filename):
    # Path of the model where model is stored
    path_to_model = r'model/model_inceptionV3_epoch5.h5'
    print("Loading the model..")

    # Load model using load_model function of Keras
    model = keras.models.load_model(path_to_model, compile=False)
    model.compile()
    print("Done!")

    img_ = tf.keras.utils.load_img(filename, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0)
    img_processed /= 255.

    # prediction using already loaded model
    prediction = model.predict(img_processed)

    index = np.argmax(prediction)

    print(category[index])

    return category[index]
