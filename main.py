import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle


import Project1 as P1
import Project2 as P2
import Project3 as P3
import Project4 as P4



hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

st.title("Welcome To Rupesh Dubey's - Web App!!!")

Projectlist=[]
Projectlist.append('Predication - Linear Regression')
Projectlist.append('Predication - RandomForest Regressor')
Projectlist.append('Classification - Decision Tree')
Projectlist.append('Classification System - Multi Algorithms')


if st.checkbox("Projects", key=12):
    Project = st.radio(
        "Select the Project",
        (Projectlist))

    if Project == 'Predication - Linear Regression': P1.Pro1()
    if Project == 'Predication - RandomForest Regressor': P2.Pro2()
    if Project == 'Classification - Decision Tree': P3.Pro3()
    if Project == 'Classification System - Multi Algorithms': P4.Pro4()


# Create a button, that when clicked, shows a text
if (st.button("About Me")):
    col11, col12 = st.columns(2)
    with col11:
        pic="https://media-exp1.licdn.com/dms/image/C4D03AQGcObyFZvfRtQ/profile-displayphoto-shrink_400_400/0/1645786089098?e=1654732800&v=beta&t=Xg4P7ieCVtwf5f_H0vIie8TRdVbR2eMdJbqi2bfWOZQ"
        st.image(pic, caption="Me", output_format="auto")
    with col12:
        with st.container():
            st.subheader("Hello, welcome to my first web application created using Streamlit on Python.\n Here, "
                         "I will posting some of learnings in Analytics field on the go. For example last week I learnt about Regression and"
                         "then about how about presenting it to the users. From that I got the idea of creating this app online. "
                         "Below are my few online certification done in this field. Also, I have added few projects to test as well. "
                         "Feel free to explore and play around and do share your feedback.")


    st.subheader("Certificates")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        image="https://s3.amazonaws.com/coursera_assets/meta_images/generated/CERTIFICATE_LANDING_PAGE/CERTIFICATE_LANDING_PAGE~7FLA7JPYU273/CERTIFICATE_LANDING_PAGE~7FLA7JPYU273.jpeg"
        st.image(image, caption="Python for Data Science, AI & Development", output_format="auto")
    with col2:
        image="https://s3.amazonaws.com/coursera_assets/meta_images/generated/CERTIFICATE_LANDING_PAGE/CERTIFICATE_LANDING_PAGE~DZSE9773S8A2/CERTIFICATE_LANDING_PAGE~DZSE9773S8A2.jpeg"
        st.image(image, caption="SQL for Data Science", output_format="auto")
    with col3:
        image="https://s3.amazonaws.com/coursera_assets/meta_images/generated/CERTIFICATE_LANDING_PAGE/CERTIFICATE_LANDING_PAGE~9CLH6FXWBB3G/CERTIFICATE_LANDING_PAGE~9CLH6FXWBB3G.jpeg"
        st.image(image, caption="Data Visualization and Communication with Tableau", output_format="auto")
    with col4:
        image="https://s3.amazonaws.com/coursera_assets/meta_images/generated/CERTIFICATE_LANDING_PAGE/CERTIFICATE_LANDING_PAGE~NAJL962VEGM5/CERTIFICATE_LANDING_PAGE~NAJL962VEGM5.jpeg"
        st.image(image, caption="Basic Statistics", output_format="auto")
    with col5:
        image="https://s3.amazonaws.com/coursera_assets/meta_images/generated/CERTIFICATE_LANDING_PAGE/CERTIFICATE_LANDING_PAGE~DFU5L2ABS8TD/CERTIFICATE_LANDING_PAGE~DFU5L2ABS8TD.jpeg"
        st.image(image, caption="Business Metrics for Data-Driven Companies", output_format="auto")
    with col6:
        image="https://s3.amazonaws.com/coursera_assets/meta_images/generated/CERTIFICATE_LANDING_PAGE/CERTIFICATE_LANDING_PAGE~THW33CM8UBUH/CERTIFICATE_LANDING_PAGE~THW33CM8UBUH.jpeg"
        st.image(image, caption="Tools for Data Science", output_format="auto")

    # st.balloons()


