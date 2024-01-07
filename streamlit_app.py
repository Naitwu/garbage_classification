import streamlit as st
import numpy as np
import pandas as pd
#import pydeck as pdk
from PIL import Image
from inference import infer
import time


st.set_page_config(page_icon="", page_title="Garbage classification")


def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://i.imgur.com/8ec0YYc.jpeg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()





st.markdown("<h1 style='text-align: center; color: orange;'>Garbage classification (🇮🇳 SeeFood)</h1>", unsafe_allow_html=True)

st.markdown("<h4 style='text-align: center; color: orange;'>A EfficientNetV2B2 neural network trained to recognize 7 types of garbage. </a> </h4>", unsafe_allow_html=True)

st.markdown("<h4 style='text-align: center; color: orange;'>We present the top five predictions .</h4>", unsafe_allow_html=True)


col1, col2, col3 = st.columns([1,18,1])


img = 0

def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image




# Uploading the File to the Page
uploadFile = st.file_uploader(label="Upload image", type=['jpg', 'png','jpeg'])

# with col1:
#         st.write(' ')


with col2:



    if uploadFile is not None:


        img = load_image(uploadFile)


        with st.spinner('Wait for it...'):

            list_of_predictions,pred_prob = infer(uploadFile)
            st.image(img)
            time.sleep(1)


        list_of_predictions = [i.replace('_',' ') for i in list_of_predictions]


        st.markdown("<p style='text-align: center; color: white;'>The most likely garbage types are as follows 👇.</p>", unsafe_allow_html=True)

        if pred_prob[0] > 0.0:

            st.markdown("""
  #### <p style="color:white">* This garbage is {}, with a probability of {:.2f} .</p>
""".format(list_of_predictions[0],pred_prob[0],),  unsafe_allow_html=True)



        if pred_prob[1] > 0 :

            st.markdown("""
  #### <p style="color:white">* This garbage is {}, with a probability of {:.2f} .</p>
""".format(list_of_predictions[1],pred_prob[1],),  unsafe_allow_html=True)


        if pred_prob[2] > 0:

            st.markdown("""
  #### <p style="color:white">* This garbage is {}, with a probability of {:.2f} .</p>
""".format(list_of_predictions[2],pred_prob[2],),  unsafe_allow_html=True)


        if pred_prob[3] > 0:

            st.markdown("""
  #### <p style="color:white">* This garbage is {}, with a probability of {:.2f} .</p>
""".format(list_of_predictions[3],pred_prob[3],),  unsafe_allow_html=True)


        if pred_prob[4] > 0:

            st.markdown("""
  #### <p style="color:white">* This garbage is {}, with a probability of {:.2f} .</p>
""".format(list_of_predictions[4],pred_prob[4],),  unsafe_allow_html=True)










        #st.write("Image Uploaded Successfully")
        st.markdown("<p style='text-align: center; color: white;'>Image Uploaded Successfully</p>", unsafe_allow_html=True)
    else:
        #st.write("Make sure you image is in JPG/PNG Format.")

        st.markdown("<p style='text-align: center; color: white;'>Make sure you image is in JPG/PNG Format.</p>", unsafe_allow_html=True)







