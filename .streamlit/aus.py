import streamlit as st
import pandas as pd
from neuralprophet import NeuralProphet
import matplotlib.pyplot as plt

#Read the csv file
data = pd.read_csv("weatherAUS.csv")
print(data)
print(data.columns)
print(data.dtypes)

#Get the specific location (Melbourne)
Melb_data = data[data["Location"]=="Melbourne"]
print(Melb_data)

#Converting Date(object) to Date(Datetime)
Melb_data["Date"] = pd.to_datetime(Melb_data["Date"])
print(Melb_data.dtypes)

plt.plot(Melb_data["Date"], Melb_data["Temp3pm"])
plt.title("Yearly Temperature at 3PM")
plt.xlabel("Year")
plt.ylabel("Temperature at 3PM")
plt.show()

#Analysis: There are missing data of temperature in 2015 onwards

#Created a new column called "Year"
Melb_data["Year"] = Melb_data["Date"].apply(lambda x: x.year)

#Filtering Year by removing greater than 2015
Melb_data = Melb_data[Melb_data["Year"]<=2015]
plt.plot(Melb_data["Date"], Melb_data["Temp3pm"])
plt.title("Yearly Temperature at 3PM")
plt.xlabel("Year")
plt.ylabel("Temperature at 3PM")
plt.show()

print(Melb_data.head())
print(Melb_data.isnull().sum())

#Create a new dataset containing date and temperature at 3PM only. 
f_data = Melb_data[["Date", "Temp3pm"]]
print(f_data)
print(f_data.isnull().sum())
print(f_data.shape)

#Change the column name for time series
f_data.columns = ["ds", "y"]
print(f_data)

#-------------------streamlit content-----------------#
st.title("Melbourne Australia Temperature Prediction at 3PM")
st.image("https://www.tide-forecast.com/tidelocationmaps/Melbourne-Australia.10.gif")

period = st.sidebar.slider("Period", min_value=0, max_value=1500)
st.sidebar.write(f"Adjusted Period: {period}")


def predict():
    #Train the model
    model = NeuralProphet()

    #frequency set to Daily 
    model.fit(f_data, freq="D")

    #Forecasting
    future= model.make_future_dataframe(f_data, periods=int(period))
    forecast = model.predict(future)
    st.header("Prediction")
    print(forecast.head())
    print(forecast.tail())

    f_model = model.plot(forecast)
    st.subheader("Forecast Plot")
    st.pyplot(f_model)

    plot_comp = model.plot_components(forecast)
    st.subheader("Trends, Yearly and Weekly Seasonal Patterns")
    st.pyplot(plot_comp)

predict_button = st.sidebar.button("Predict Certain Period")

if predict_button == True:
    predict()
