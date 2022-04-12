import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle

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

if st.checkbox("Projects"):
    Project = st.radio(
        "Select the Project",
        ('Linear Regression', 'RandomForest Regressor', 'Decision Tree'))
    if Project == 'Linear Regression':
        st.write(
            "This a ML model for predicting flight price using Multi Linear Regression on Kaggle dataset.")
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
            st.dataframe(data)
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

    if Project == 'RandomForest Regressor':
        st.write("This a ML model for predicting car price using Random Forest regressor with Hyperparameter Tuning on Car Dekho dataset")

        df = pd.read_csv(r'car data.csv')
        final_df = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
                       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
        # Creating new column for driving year old cars
        final_df['No_year_old'] = 2022 - final_df['Year']

        st.title('Car Price Prediction')

        with st.sidebar:
            st.write("Select your choice.")

        Year = st.sidebar.number_input(
            'Select Year',
            min_value=2005, max_value=2022
        )
        Present_Price = st.sidebar.number_input(
            'Enter Present Value',
            min_value=0.5, max_value=final_df.Present_Price.max()
        )

        Kms_Driven = st.sidebar.number_input(
            'Enter Kms Driven',
            min_value=1000, max_value=50000
        )
        Kms_Driven2 = np.log(Kms_Driven)
        Owner = st.sidebar.selectbox(
            'Select No. of Owners',
            (final_df["Owner"].unique())
        )
        Fuel_Type = st.sidebar.selectbox(
            'Select Fuel Type',
            (["Petrol", "Diesel", "CNG"])
        )
        if (Fuel_Type == 'Petrol'):
            Fuel_Type_Petrol = 1
            Fuel_Type_Diesel = 0
        elif(Fuel_Type == 'Diesel'):
            Fuel_Type_Petrol = 0
            Fuel_Type_Diesel = 1

        Year = 2020 - Year

        Seller_Type_Individual = st.sidebar.selectbox(
            'Are you A Dealer or Individual',
            (["Individual", "Dealer"])
        )

        if (Seller_Type_Individual == 'Individual'):
            Seller_Type_Individual = 1
        else:
            Seller_Type_Individual = 0

        Transmission_Mannual = st.sidebar.selectbox(
            'Transmission type',
            (["Manual Car", "Automatic Car"])
        )
        if (Transmission_Mannual == 'Mannual'):
            Transmission_Mannual = 1
        else:
            Transmission_Mannual = 0


        st.subheader(
            f"You have selected a Car {Year} years old with present price of {Present_Price} lakhs, {Fuel_Type} version.")
        model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))
        prediction = model.predict([[Present_Price, Kms_Driven2, Owner, Year, Fuel_Type_Diesel, Fuel_Type_Petrol,
                                     Seller_Type_Individual, Transmission_Mannual]])
        output = round(prediction[0], 2)
        if output <= 0:
            st.write("Sorry you cannot sell this car")
        else:
            st.write(f"You Can Sell The Car in {output} lakhs.")



        def load_data1(nrows):
            data = pd.read_csv(r'car data.csv', nrows=nrows)
            lowercase = lambda x: str(x).lower()
            data.rename(lowercase, axis='columns', inplace=True)
            return data



        if st.checkbox('Show raw data'):
            st.subheader('Raw data')
            data = load_data1(1000)
            st.dataframe(data)
            st.write('Shape of dataset:', df.shape)

        if st.checkbox('Show me EDA'):
            st.text("Simple EDA of raw data")
            col1, col2, col3 = st.columns(3)

            with col1:
                bar_df = df.Fuel_Type.value_counts()
                st.bar_chart(bar_df,width=10, height=200)
            with col2:
                bar_df = df.Transmission.value_counts()
                st.bar_chart(bar_df,width=10, height=200)
            with col3:
                bar_df = df.Seller_Type.value_counts()
                st.bar_chart(bar_df,width=10, height=200)

    if Project == 'Decision Tree':
        st.write(
            "This a ML model for classifying safety of car using Decision tree algorithm on demo dataset.")
        dfc = pd.read_csv(r'car_evaluation.csv',header=None)
        col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
        dfc.columns = col_names

        XDT = dfc.drop(['class'], axis=1)
        ydt = dfc['class']

        # split X and y into training and testing sets

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(XDT, ydt, test_size=0.33, random_state=42)
        X_train1 = X_train.copy(deep=True)
        X_test1 = X_test.copy(deep=True)

        buying_dict = {'vhigh': 1, 'high': 2, 'med': 3, 'low': 4}
        maint_dict = {'vhigh': 1, 'high': 2, 'med': 3, 'low': 4}
        doors_dict = {'2': 2, '3': 3, '4': 4, '5more': 5}
        persons_dict = {'2': 2, '4': 4, 'more': 5}
        lug_boot_dict = {'small': 1, 'big': 3, 'med': 2}
        safety_dict = {'low': 1, 'high': 3, 'med': 2}

        X_train1.buying = X_train1.buying.map(buying_dict)
        X_train1.maint = X_train1.maint.map(maint_dict)
        X_train1.doors = X_train1.doors.map(doors_dict)
        X_train1.persons = X_train1.persons.map(persons_dict)
        X_train1.lug_boot = X_train1.lug_boot.map(lug_boot_dict)
        X_train1.safety = X_train1.safety.map(safety_dict)

        X_test1.buying = X_test1.buying.map(buying_dict)
        X_test1.maint = X_test1.maint.map(maint_dict)
        X_test1.doors = X_test1.doors.map(doors_dict)
        X_test1.persons = X_test1.persons.map(persons_dict)
        X_test1.lug_boot = X_test1.lug_boot.map(lug_boot_dict)
        X_test1.safety = X_test1.safety.map(safety_dict)

        st.title('Car Safety (Decision Tree)')

        with st.sidebar:
            st.write("Select your choice.")

        buying = st.sidebar.selectbox(
            'Select Buying Category',
            (dfc.buying.unique())
        )
        maint = st.sidebar.selectbox(
            'Select Maintenance Category',
            (dfc.maint.unique())
        )
        doors = st.sidebar.selectbox(
            'Select Doors',
            (dfc.doors.unique())
        )

        persons = st.sidebar.selectbox(
            'Select Person Capacity',
            (dfc.persons.unique())
        )

        lug_boot = st.sidebar.selectbox(
            'Select Luggage size',
            (dfc.lug_boot.unique())
        )

        safety = st.sidebar.selectbox(
            'Select Safety class',
            (dfc.safety.unique())
        )


        list1 = []
        list1.append(buying)
        list1.append(maint)
        list1.append(doors)
        list1.append(persons)
        list1.append(lug_boot)
        list1.append(safety)
        dfw = pd.DataFrame([list1])
        col_names1 = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
        dfw.columns = col_names1

        dfw.buying = dfw.buying.map(buying_dict)
        dfw.maint = dfw.maint.map(maint_dict)
        dfw.doors = dfw.doors.map(doors_dict)
        dfw.persons = dfw.persons.map(persons_dict)
        dfw.lug_boot = dfw.lug_boot.map(lug_boot_dict)
        dfw.safety = dfw.safety.map(safety_dict)

        # import DecisionTreeClassifier

        from sklearn.tree import DecisionTreeClassifier

        # instantiate the DecisionTreeClassifier model with criterion entropy
        clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)

        # fit the model
        clf_en.fit(X_train1, y_train)
        # Predict the Test set results with criterion entropy
        y_pred_en = clf_en.predict(X_test1)
        # Check accuracy score with criterion entropy
        from sklearn.metrics import accuracy_score

        st.write('Model accuracy score with criterion entropy: {0:0.4f}'.format(accuracy_score(y_test, y_pred_en)))

        prediction = clf_en.predict(dfw)
        st.subheader(f"Your Car's Safety evaluation is in {prediction} class.")

        def load_data3(nrows):
            data = dfc
            lowercase = lambda x: str(x).lower()
            data.rename(lowercase, axis='columns', inplace=True)
            return data


        if st.checkbox('Show raw data'):
            st.subheader('Raw data')
            data = load_data3(1000)
            st.dataframe(data)
            st.write('Shape of dataset:', dfc.shape)

        if st.checkbox('Show Tree diagram', key=12):
            from sklearn import tree

            fig = plt.figure(figsize=(10, 4))
            ax1 = tree.plot_tree(clf_en.fit(X_train1, y_train))
            st.pyplot(fig)



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
