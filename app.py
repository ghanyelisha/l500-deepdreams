# Native imports
import os
import argparse
import math
import numbers
from collections import namedtuple, defaultdict
import enum

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms

# Visualize imports
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# App related
import streamlit as st
from predict import *
from io import BytesIO

# Parameters for the model
config = {
    "model_name": 'RESNET50',
    "pretrained_weights": 'IMAGENET_V1',
    "pyramid_size": 3,
    "pyramid_ratio": 2.1,
    "num_gradient_ascent_iterations": 12,
    "lr": 0.05,
    "img_width" : 800,
    "input" : "data/input/test.jpg",
    "layers_to_use" : ["layer3"],
    "should_display" : False,
    "spatial_shift_size" : 30,
    "smoothing_coefficient" : 0.4,
    "use_noise " : False}

config['dump_dir'] = os.path.join(OUT_IMAGES_PATH, f'{config["model_name"]}_{config["pretrained_weights"]}')
config['input'] = os.path.basename(config['input'])  # handle absolute and relative paths

# Markdown
st.title("Group DeepDream Project")
st.write("### From Ghany Elisha (ghannyelisha@gmail.com)")
st.write("##### Connect on GitHub: https://github.com/ghanyelisha")

st.write("#")

st.write("Dive into a kaleidoscope of AI wonder with DeepDream - the mind-bending art generator that dances on the edge of reality and imagination! Picture this: a neural network unlike any other, craving artistic expression!")

st.write("#####")

st.write(" DeepDream takes ordinary images and transforms them into captivating masterpieces that tease the boundaries of the possible. Watch as it conjures mesmerizing patterns, morphs everyday scenes into dazzling dreamscapes, and leaves you wondering if you've stumbled into a digital wonderland.")

st.write("#####")


st.write(":red[EXAMPLES OF GENERATED IMAGES]")
col1, col2 = st.columns(2)
with col1:
   st.image("data/samples/deepdream_sample1.jpg")
with col2:
   st.image("data/samples/deepdream_sample3.jpg")

st.subheader("Upload your Image below")

st.write("#####")

uploaded_file = st.file_uploader("Only JPG, PNG, and JPEG formats supported as of today...", type=['jpg', 'png'])

if uploaded_file is not None:
    

    temp_img = Image.open(uploaded_file)
    input_image = np.array(temp_img)
    config['img_width'] = input_image.shape[1]
            
    # Float 32 and in range [0, 1]
    input_image = input_image.astype(np.float32)
    input_image /= 255.0
    
    #st.write(input_image)
    st.write("### Uploaded Image looks like this")
    st.image(input_image, channels = "RGB")

    st.write("Received Image succesfully")
    st.write("Uploaded image width is", input_image.shape[1], "pixels")
    #st.write("Type of Input Image", type(input_image))

    # ALL CONFIG VARIABLES
    st.write("#####")
    st.subheader("Customization Options Available")
    st.write("###### :red[Leave at default values if you are overwhelmed as they work  the best.] ")
    st.write("#")
    
    image_width = st.slider("Image Width (aspect ratio will be preserved)", 400, 700, 600, help = 'Higher resolution might take time to process')
    config['img_width'] = image_width
    target_shape = config["img_width"]
    if target_shape is not None:
        if isinstance(target_shape, int) and target_shape != -1:
            curr_height, curr_width = input_image.shape[0], input_image.shape[1]
            new_width = target_shape
            new_height = int(curr_height * (new_width / curr_width))
            #print(type(new_height))
            input_image = cv.resize(input_image, (new_width, new_height), interpolation = cv.INTER_CUBIC)
        else:
            input_image = cv.resize(input_image, (target_shape[1], target_shape[0]), interpolation = cv.INTER_CUBIC)
    # Got the new input image width
    
    layers_to_use = ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']
    selected_layer = st.selectbox("Which layer of the RESNET50 network to use", layers_to_use, index = 2, help = 'Layer 3 gived the best results but feel free to use other layers')
    config['layers_to_use'] = [selected_layer]
    
    dataset_version = ['IMAGENET_V1', 'IMAGENET_V2']
    selected_version = st.selectbox("Which Dataset to use", dataset_version, index = 1, help = 'Adding support for more datasets in the future')
    
    how_many_octaves = st.slider("Number of Octaves to use", 2, 7, 5, help = 'Higher the octaves, more variations in the size of the features generated')
    config["pyramid_size"] = how_many_octaves
    
    pyramid_ratio = st.slider("Octave Ratio", 1.2, 1.8, 1.5, step = 0.1, help = 'The factor by which the octave reduces in size')
    config['pyramid_ratio'] = pyramid_ratio
    
    smoothing = st.slider("Smoothing factor", 0.2, 0.7, 0.4, step = 0.1, help = 'Higher the value more blurred the original features are')
    config["smoothing_coefficient"] = smoothing
    
    lr = st.slider("Learning Rate", 0.01, 0.10, 0.08, step = 0.01, help = "Higher the learning rate, faster it will generate but might not produce good results")
    config["lr"] = lr
    
    iterations = st.slider("Number of Gradient Ascent Iterations", 1, 15, 5, help = "Higher the iterations more enhanced the features are")
    config["num_gradient_ascent_iterations"] = iterations
    
    if st.button("Click here to generate!"):
        st.write("Generating")
        with st.spinner("AI is Dreaming"):
            count = 0
            while True:
                output_image = deep_dream_static(config, input_image)
                count += 1
                cv.imwrite("data/out-images/" + str(count) + ".jpg", output_image[:, : , ::-1] * 255)
                break


        st.write("# Generated Output")
        st.image(output_image, channels = "RGB")
        
        # buf = BytesIO()
        # downloaded_image = Image.fromarray(input_image.reshape((input_image.shape[0], input_image.shape[1], 3)))
        # downloaded_image.save(buf, format="JPEG")
        # byte_im = buf.getvalue()
        
        
        # Download
        st.download_button(
            label="Download Generated image for free",
            data = open("data/out-images/" + str(count) + ".jpg", "rb").read(),
            file_name = "DeepDream.jpg",
            mime="image/jpg",
        )
        
        # st.write("##### Feature coming in the future -\n1) Generating video from an input image.\n2) Support for more datasets and networks.")
        
        st.write("#")
        st.write("#")
        
        st.write("### From Ghany Elisha (ghannyelisha@gmail.com)")
        st.write("##### Connect on GitHub: https://github.com/ghanyelisha")
        

# if uploaded_file is None:
#     del input_image
#     del output_image