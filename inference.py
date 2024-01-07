import tensorflow as tf
from tensorflow import keras
import numpy as np
#from keras.preprocessing import image

from model import model_def

import pandas as pd


import json

from tensorflow.keras.utils import load_img, img_to_array


def infer(img):


    img_width, img_height = 224, 224
    img = load_img(img, target_size = (img_width, img_height))
    img = img_to_array(img)
    img = np.expand_dims(img, axis = 0)/255.
    ## Create Model Object

    model = model_def()

    # Loads the weights
    model.load_weights("training_1/cp.ckpt")

    predicted_vector = model.predict(img,use_multiprocessing=True)

    #print(predicted_vector.shape)

    ## Convert the vector to list

    predicted_vector_list = predicted_vector.tolist()



    predicted_vector_list = sum(predicted_vector_list,[])

    ### Getting top 3 indices

    def sort_index(lst, rev=True):
        index = range(len(lst))
        s = sorted(index, reverse=rev, key=lambda i: lst[i])
        return s


    top_five= sort_index(predicted_vector_list)[:5]

    #print(top_five)
    with open('class_mappings.json') as json_file:
        data = json.load(json_file)

    data = {y: x for x, y in data.items()}

    list_of_predictions = []

    for index in top_five:
        list_of_predictions.append(data[index])

    top_five_values_prediction = sorted(predicted_vector_list,reverse=True)[:5]


    return(list_of_predictions, top_five_values_prediction)
