import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Titanic Təxmini", page_icon="🚢")
st.title("🚢 Titanic Sağ Qalma Təxmini")

@st.cache_resource
def load_model():
    with open('titanic_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

st.sidebar.header("Sərnişin Məlumatları")
pclass = st.sidebar.selectbox("Bilet Sinfi", [1, 2, 3], index=2)
sex = st.sidebar.radio("Cinsiyyət", ["Qadın", "Kişi"])
age = st.sidebar.slider("Yaş", 1, 80, 28)
fare = st.sidebar.slider("Bilet Qiyməti ($)", 0.0, 500.0, 32.0)
family_size = st.sidebar.slider("Ailə Sayı", 1, 10, 1)
embarked = st.sidebar.selectbox("Liman", ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"])

sex_val = 1 if sex == "Qadın" else 0
embarked_Q = 1 if "Q" in embarked else 0
embarked_S = 1 if "S" in embarked else 0

input_data = np.array([[pclass, sex_val, age, fare, family_size, embarked_Q, embarked_S]])

if st.button("Təxmin Et 🔮"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]
    
    if prediction == 1:
        st.success(f"🎉 **Sağ Qaldı!** (Ehtimal: {proba*100:.1f}%)")
    else:
        st.error(f"☠️ **Həlak Oldu.** (Sağ qalma ehtimalı: {proba*100:.1f}%)")
