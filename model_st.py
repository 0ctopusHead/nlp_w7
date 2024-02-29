import streamlit as st
import pickle
import warnings
from PIL import Image

warnings.filterwarnings("ignore")
pickle_in = open("model/model_iris.pkl", "rb")
classifier = pickle.load(pickle_in)


def predict_iris_variety(sepal_length, sepal_width, petal_length, petal_width):
    prediction = classifier.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    print(prediction)
    return prediction

def Input_Output():
    st.title("Iris variety Prediction")
    st.image("https://zahidhasan.github.io/images/iris.png", width=600)

    st.markdown("You're using Streamlit...", unsafe_allow_html=True)
    sepal_length = st.text_input("Enter Sepal Length", ".")
    sepal_width = st.text_input("Enter Sepal Width", ".")
    petal_length = st.text_input("Enter Petal Length", ".")
    petal_width = st.text_input("Enter Petal Width", ".")

    result =""
    if st.button("Click here to Predict"):
        result = predict_iris_variety(sepal_length,sepal_width,petal_length,petal_width)
        st.balloons()
    st.success("The output is {}".format(result))


if __name__ == "__main__":
    Input_Output()