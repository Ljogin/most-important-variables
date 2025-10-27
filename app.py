import streamlit as st
import pandas as pd
from pycaret.classification import setup as cls_setup, compare_models as cls_compare, pull as cls_pull, plot_model as cls_plot
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, pull as reg_pull, plot_model as reg_plot

# ==========================
# CONFIGURACJA STRONY
# ==========================
st.set_page_config(page_title="AutoML Analyzer", layout="wide")
st.title("🤖 AutoML Analyzer – automatyczna analiza danych z PyCaret")
st.markdown("""
Aplikacja umożliwia:
1. Wczytanie pliku CSV  
2. Wybór kolumny docelowej  
3. Automatyczne rozpoznanie typu problemu  
4. Budowę najlepszego modelu ML  
5. Wyświetlenie najważniejszych cech  
6. Wygenerowanie słownego opisu wyników  
""")

# ==========================
# v1 – Wczytanie pliku CSV
# ==========================
uploaded_file = st.file_uploader("📂 Wgraj plik CSV", type=["csv"])
if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Błąd podczas wczytywania pliku: {e}")
        st.stop()

    st.success(f"✅ Załadowano plik: {uploaded_file.name}")
    st.write("### Podgląd danych:")
    st.dataframe(data.head())
else:
    st.info("➡️ Załaduj plik CSV, aby rozpocząć.")
    st.stop()

# ==========================
# v2 – Wybór kolumny docelowej
# ==========================
target = st.selectbox("🎯 Wybierz kolumnę docelową (y):", options=data.columns)

# ==========================
# v3 – Rozpoznanie typu problemu
# ==========================
if pd.api.types.is_numeric_dtype(data[target]):
    # jeśli kolumna numeryczna, ale ma mało unikalnych wartości → klasyfikacja
    if len(data[target].unique()) <= 10:
        problem_type = "classification"
    else:
        problem_type = "regression"
else:
    problem_type = "classification"

st.markdown(f"### 🔍 Wykryty typ problemu: **{problem_type.upper()}**")

# ==========================
# v4 – Budowa modelu ML
# ==========================
st.subheader("⚙️ Trenowanie modelu AutoML")

if st.button("🚀 Uruchom AutoML"):
    with st.spinner("Trwa analiza i budowa najlepszego modelu..."):
        try:
            if problem_type == "classification":
                cls_setup(data=data, target=target, silent=True, session_id=42)
                best_model = cls_compare()
                results = cls_pull()
            else:
                reg_setup(data=data, target=target, silent=True, session_id=42)
                best_model = reg_compare()
                results = reg_pull()
        except Exception as e:
            st.error(f"Błąd podczas trenowania modelu: {e}")
            st.stop()

    st.success("✅ Model został pomyślnie zbudowany!")
    st.write("### 📊 Wyniki porównania modeli:")
    st.dataframe(results)
    st.write("### 🏆 Najlepszy model:", type(best_model).__name__)

    # ==========================
    # v5 – Wyświetlenie najważniejszych cech
    # ==========================
    st.subheader("🌟 Najważniejsze cechy modelu")
    try:
        if problem_type == "classification":
            cls_plot(best_model, plot="feature")
        else:
            reg_plot(best_model, plot="feature")
        st.pyplot(bbox_inches="tight")
    except Exception:
        st.info("⚠️ Model nie udostępnia wykresu ważności cech lub dane są nieodpowiednie.")

    # ==========================
    # v6 – Opis słowny wyników
    # ==========================
    st.subheader("📝 Opis analizy i rekomendacje")
    st.markdown(f"""
    **Zadanie:** {problem_type.capitalize()}  
    **Kolumna docelowa:** `{target}`  
    **Najlepszy model:** `{type(best_model).__name__}`  

    **Wnioski:**
    - Aplikacja automatycznie wybrała model o najlepszym wyniku spośród testowanych przez PyCaret.
    - Wykres powyżej pokazuje, które cechy mają największy wpływ na przewidywania.
    - Aby poprawić jakość modelu:
        - sprawdź dane wejściowe pod kątem braków i wartości odstających,  
        - usuń zbędne kolumny lub dodaj nowe dane,  
        - wypróbuj inny zestaw cech lub większy zbiór danych.  

    Model można dalej ulepszać, korzystając z pełnych funkcji PyCaret lub eksportując model do pliku `.pkl`.
    """)
else:
    st.info("Kliknij przycisk powyżej, aby uruchomić proces AutoML.")

