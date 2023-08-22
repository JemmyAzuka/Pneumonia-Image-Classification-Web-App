import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

from util import classify, set_background



image_directory = "C:/Users/Lenovo/Desktop/Dual Disease Prediction System/images/icons8.png"
image = Image.open(image_directory)
PAGE_CONFIG = {"page_title":"Pneumonia Image Classification Web App", "page_icon":image, "layout":"centered", "initial_sidebar_state":"auto"}
st.set_page_config(**PAGE_CONFIG)




set_background('C:/Users/Lenovo/AppData/Local/Programs/Microsoft VS Code/Pneumonia Image Classification Web App/bgs/ip.png')

# set title
st.title('Pneumonia Classification')

# set header
st.header('Please upload a chest X-ray image')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model('C:/Users/Lenovo/AppData/Local/Programs/Microsoft VS Code/Pneumonia Image Classification Web App/model/pneumonia_classifier.h5')

# load class names
with open('C:/Users/Lenovo/AppData/Local/Programs/Microsoft VS Code/Pneumonia Image Classification Web App/model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))
