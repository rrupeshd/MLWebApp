import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px

st.set_page_config(
   page_title="Rupesh Dubey - Web App",
   page_icon="🧊",
   layout="wide",
   initial_sidebar_state="expanded",
)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)



st.title("Welcome To Rupesh Dubey's First Web App!!! - ")

# Create a button, that when clicked, shows a text
if (st.button("About Me")):

    st.header("Certificates")
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

    st.snow()

with st.expander("Projects"):
#if (st.button("Projects")):
    if st.checkbox('Flight Price Prediction'):
        clean_df = pd.read_csv(r'Clean_Dataset.csv')
        del clean_df["Unnamed: 0"]
        x_df = clean_df.copy(deep=True)
        x_df.drop(['flight', 'price'], axis=1, inplace=True)
        y_df = clean_df["price"].copy(deep=True)

        airline_dict = {'SpiceJet': 1, 'AirAsia': 2, 'Vistara': 3, 'GO_FIRST': 4, 'Indigo': 5, 'Air_India': 6}
        source_city_dict = {'Delhi': 1, 'Mumbai': 2, 'Bangalore': 3, 'Kolkata': 4, 'Hyderabad': 5, 'Chennai': 6}
        departure_time_dict = {'Evening': 4, 'Early_Morning': 1, 'Morning': 2, 'Afternoon': 3, 'Night': 5, 'Late_Night': 6}
        stops_dict = {'zero': 1, 'one': 2, 'two_or_more': 3}
        arrival_time_dict = {'Evening': 4, 'Early_Morning': 1, 'Morning': 2, 'Afternoon': 3, 'Night': 5, 'Late_Night': 6}
        destination_city_dict = {'Delhi': 1, 'Mumbai': 2, 'Bangalore': 3, 'Kolkata': 4, 'Hyderabad': 5, 'Chennai': 6}
        class_dict = {'Economy': 1, 'Business': 2}

        x_df.airline = x_df.airline.map(airline_dict)
        x_df.source_city = x_df.source_city.map(source_city_dict)
        x_df.departure_time = x_df.departure_time.map(departure_time_dict)
        x_df.stops = x_df.stops.map(stops_dict)
        x_df.arrival_time = x_df.arrival_time.map(arrival_time_dict)
        x_df.destination_city = x_df.destination_city.map(destination_city_dict)
        x_df["class"] = x_df["class"].map(class_dict)

        x_mapped_df = x_df.copy()
        x_mapped_df.drop(['duration'], axis=1, inplace=True)

        st.title('Flight Price Prediction')

        with st.sidebar:
            st.write("Select your choice.")

        airline_name = st.sidebar.selectbox(
            'Select Airline',
            (clean_df.airline.unique())
        )
        source_city_name = st.sidebar.selectbox(
            'Select Source City',
            (clean_df.source_city.unique())
        )
        destination_city_name = st.sidebar.selectbox(
            'Select Destination City',
            (clean_df.destination_city.unique())
        )

        departure_time_name = st.sidebar.selectbox(
            'Select Departure Time',
            (clean_df.departure_time.unique())
        )

        arrival_time_name = st.sidebar.selectbox(
            'Select Arrival Time',
            (clean_df.arrival_time.unique())
        )

        stops_name = st.sidebar.selectbox(
            'Select No. of Stops',
            (clean_df.stops.unique())
        )

        class_name = st.sidebar.selectbox(
            'Select Class',
            (clean_df["class"].unique())
        )

        Days = st.sidebar.slider('Days to travel', 1, 10, step=1)
        params = Days

        st.subheader(
            f"You have selected {airline_name} airlines from {source_city_name} to {destination_city_name} in {departure_time_name} for {class_name} class.")


        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train1, X_test1, y_train1, y_test1 = train_test_split(x_mapped_df, y_df, test_size=0.33, random_state=42)

        # Fitting Multiple Linear Regression to the Training set
        from sklearn.linear_model import LinearRegression
        model1 = LinearRegression()
        model1.fit(X_train1, y_train1)

        # Predicting the Test set results
        y_pred1 = model1.predict(X_test1)

        # Calculating the R squared value
        from sklearn.metrics import r2_score
        score=r2_score(y_test1, y_pred1)


        X_testlist=[airline_dict[airline_name], source_city_dict[source_city_name], departure_time_dict[departure_time_name], stops_dict[stops_name], arrival_time_dict[arrival_time_name], destination_city_dict[destination_city_name], class_dict[class_name],params]
        x_new_df=x_mapped_df[0:2].copy(deep=True)
        x_new_df.iloc[0] = X_testlist

        y_pred2 = model1.predict(x_new_df[0:1])
        predicated_price=str(y_pred2)[1:-1]
        predicated_price=float(predicated_price)
        st.title("Your estimated flight price is Rs. " + str(int(predicated_price)))


        st.write("R squared value is :" + str(score))



        def load_data(nrows):
            data = pd.read_csv(r'Clean_Dataset.csv', nrows=nrows)
            lowercase = lambda x: str(x).lower()
            data.rename(lowercase, axis='columns', inplace=True)
            return data



        if st.checkbox('Show raw data'):
            st.subheader('Raw data')
            data = load_data(10000)
            st.write(data)
            st.write('Shape of dataset:', clean_df.shape)

        if st.checkbox('Show me EDA'):
            st.text("Simple EDA of raw data")

            bar_df=clean_df.airline.value_counts()
            st.bar_chart(bar_df)
            airline_avg=clean_df.groupby(['airline'])['price'].mean()
            airline_avg=airline_avg.to_frame()
            st.area_chart(airline_avg)

            st.header("Airline Price from source city")
            fig = plt.figure(figsize=(10, 4))
            ax=sns.countplot(x="airline",data=clean_df,order=clean_df.airline.value_counts().index)
            ax.bar_label(ax.containers[0])
            ax.set_xlabel("Top Airlines")
            ax.set_ylabel("Number of flights")

            st.pyplot(fig)




