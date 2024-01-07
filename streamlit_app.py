import streamlit as st
import numpy as np
import pandas as pd
#import pydeck as pdk
from PIL import Image
from inference import infer
import time


st.set_page_config(page_icon="", page_title="垃圾分類")


def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://i.imgur.com/RDQcYlH.png");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()





st.markdown("<h1 style='text-align: center; color: darkblue;'>垃圾分類</h1>", unsafe_allow_html=True)

st.markdown("<h4 style='text-align: center; color: darkblue;'>使用 EfficientNetV2B1 辨識七類垃圾. </a> </h4>", unsafe_allow_html=True)

st.markdown("<h4 style='text-align: center; color: darkblue;'>提供最相似前五類與機率 .</h4>", unsafe_allow_html=True)


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

classmap_chi = {'cardboard': '紙箱', 'glass': '玻璃', 'metal': '金屬', 'paper': '紙類', 'plastic': '塑膠', 'tetra pak': '鋁箔包', 'trash': '一般垃圾'}

with col2:



    if uploadFile is not None:


        img = load_image(uploadFile)


        with st.spinner('Wait for it...'):

            list_of_predictions,pred_prob = infer(uploadFile)
            st.image(img)
            time.sleep(1)


        list_of_predictions = [i.replace('_',' ') for i in list_of_predictions]


        st.markdown("<p style='text-align: center; color: #0A0A0A;'>最可能的垃圾類型如下 👇.</p>", unsafe_allow_html=True)

        if pred_prob[0] > 0.0:

            st.markdown("""
  #### <p style="color:#0A0A0A">* 這個垃圾分類是 {}, 機率為 {:.2f} .</p>
""".format(classmap_chi[list_of_predictions[0]],pred_prob[0],),  unsafe_allow_html=True)



        if pred_prob[1] > 0 :

            st.markdown("""
  #### <p style="color:#0A0A0A">* 這個垃圾分類是 {}, 機率為 {:.2f} .</p>
""".format(classmap_chi[list_of_predictions[1]],pred_prob[1],),  unsafe_allow_html=True)


        if pred_prob[2] > 0:

            st.markdown("""
  #### <p style="color:#0A0A0A">* 這個垃圾分類是 {}, 機率為 {:.2f} .</p>
""".format(classmap_chi[list_of_predictions[2]],pred_prob[2],),  unsafe_allow_html=True)


        if pred_prob[3] > 0:

            st.markdown("""
  #### <p style="color:#0A0A0A">* 這個垃圾分類是 {}, 機率為 {:.2f} .</p>
""".format(classmap_chi[list_of_predictions[3]],pred_prob[3],),  unsafe_allow_html=True)


        if pred_prob[4] > 0:

            st.markdown("""
  #### <p style="color:#0A0A0A">* 這個垃圾分類是 {}, 機率為 {:.2f} .</p>
""".format(classmap_chi[list_of_predictions[4]],pred_prob[4],),  unsafe_allow_html=True)










        #st.write("Image Uploaded Successfully")
        st.markdown("<p style='text-align: center; color: #0A0A0A;'>Image Uploaded Successfully</p>", unsafe_allow_html=True)
    else:
        #st.write("Make sure you image is in JPG/PNG Format.")

        st.markdown("<p style='text-align: center; color: #0A0A0A;'>Make sure you image is in JPG/PNG Format.</p>", unsafe_allow_html=True)







