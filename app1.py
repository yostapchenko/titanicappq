import streamlit as st
import pickle
from datetime import datetime

# Wczytanie wcześniej wytrenowanego modelu
filename = "model.sv"

# Załaduj model w odpowiedni sposób
try:
    with open(filename, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'model.sv' is in the application directory.")
    st.stop()

# Słowniki do mapowania zakodowanych zmiennych na etykiety
pclass_d = {0: "Pierwsza", 1: "Druga", 2: "Trzecia"}
embarked_d = {0: "Cherbourg", 1: "Queenstown", 2: "Southampton"}
sex_d = {0: "Kobieta", 1: "Mężczyzna"}  # Dodano słownik płci do mapowania wartości płci

def main():
    # Konfiguracja strony aplikacji Streamlit
    st.set_page_config(page_title="Aplikacja do przewidywania przeżycia na Titanicu")

    # Kontenery dla układu strony
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    # Zmiana obrazu na bardziej adekwatny do tematyki Titanica
    st.image("https://gfx.planeta.pl/var/g3-planeta/storage/images/newsy/powstanie-titanic-ii-zbuduja-identyczny-statek/7420688-1-pol-PL/Powstanie-Titanic-II.-Buduja-statek_article.webp")

    # Sekcja przeglądowa
    with overview:
        st.title("Aplikacja do przewidywania przeżycia na Titanicu")

    # Dane wejściowe w lewej kolumnie
    with left:
        # Przycisk wyboru płci
        sex_radio = st.radio("Płeć", list(sex_d.keys()), format_func=lambda x: sex_d[x])
        
        # Przycisk wyboru klasy pasażerskiej
        pclass_radio = st.radio("Klasa Pasażerska", list(pclass_d.keys()), format_func=lambda x: pclass_d[x])
        
        # Przycisk wyboru portu zaokrętowania
        embarked_radio = st.radio("Port", list(embarked_d.keys()), format_func=lambda x: embarked_d[x])

    # Dane wejściowe w prawej kolumnie
    with right:
        # Suwak dla wieku, bazujący na oryginalnym zakresie danych (min: 0.42, max: 80)
        age_slider = st.slider("Wiek", min_value=0.42, max_value=80.0, value=29.0)
        
        # Suwak dla liczby członków rodziny, bazujący na danych (min: 0, max: 10)
        family_slider = st.slider("Liczba członków rodziny na pokładzie", min_value=0, max_value=10, value=0)
        
        # Suwak dla opłaty za przejazd, bazujący na danych (min: 0, max: 512)
        fare_slider = st.slider("Opłata za przejazd", min_value=0.0, max_value=512.0, value=32.0)

    # Sekcja przewidywania
    with prediction:
        st.subheader("Czy pasażer przeżyłby?")

        # Przygotowanie danych wejściowych do przewidywania
        input_data = [[pclass_radio, sex_radio, age_slider, family_slider, fare_slider, embarked_radio]]

        try:
            # Wykonanie przewidywania
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)

            # Wyświetlenie wyniku
            if prediction[0] == 1:
                st.success(f"Pasażer przeżyłby z prawdopodobieństwem {prediction_proba[0][1] * 100:.2f}%.")
            else:
                st.error(f"Pasażer nie przeżyłby z prawdopodobieństwem {prediction_proba[0][0] * 100:.2f}%.")
        except Exception as e:
            st.error(f"Error making prediction: {e}")

if __name__ == '__main__':
    main()
