
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="AutoML Lite", layout="wide")
st.title("🤖 AutoML Lite – automatyczna analiza danych")

uploaded_file = st.file_uploader("📂 Wgraj plik CSV", type=["csv"])
if not uploaded_file:
    st.info("⏳ Oczekuję na wgranie pliku CSV...")
    st.stop()

try:
    with st.spinner("📥 Wczytywanie danych..."):
        data = pd.read_csv(uploaded_file, sep=';', engine='python')
    st.success(f"Załadowano {uploaded_file.name} ({data.shape[0]} wierszy, {data.shape[1]} kolumn).")
    st.dataframe(data.head())
except Exception as e:
    st.error(f"❌ Błąd przy wczytywaniu pliku: {e}")
    st.stop()

if data is not None:
    target_col = st.selectbox("🎯 Wybierz kolumnę docelową:", data.columns)

    try:
        if data[target_col].dtype == "object" or len(data[target_col].unique()) < 10:
            problem_type = "classification"
        else:
            problem_type = "regression"
        st.write(f"🔍 Zidentyfikowano problem: **{problem_type.upper()}**")
    except Exception as e:
        st.error(f"❌ Błąd przy analizie kolumny docelowej: {e}")
        st.stop()

    try:
        df = data.dropna()
        if df.empty:
            st.error("❌ Dane po usunięciu braków są puste. Nie można przeprowadzić analizy.")
            st.stop()

        X = df.drop(columns=[target_col])
        y = df[target_col]

        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        if problem_type == "classification" and y.dtype == "object":
            y = LabelEncoder().fit_transform(y)

        if X.empty or len(y) == 0:
            st.error("❌ Dane wejściowe są puste po przetworzeniu. Sprawdź plik CSV.")
            st.stop()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        st.error(f"❌ Błąd przy przygotowaniu danych: {e}")
        st.stop()

    if st.button("🚀 Uruchom analizę"):
        with st.spinner("🧠 Trwa trenowanie modeli..."):
            try:
                if problem_type == "classification":
                    models = {
                        "LogisticRegression": LogisticRegression(max_iter=500),
                        "RandomForest": RandomForestClassifier()
                    }
                    metric = accuracy_score
                else:
                    models = {
                        "LinearRegression": LinearRegression(),
                        "RandomForest": RandomForestRegressor()
                    }
                    metric = r2_score

                best_model, best_score = None, -999
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    score = metric(y_test, preds)
                    if score > best_score:
                        best_model, best_score = model, score

                st.write("🔍 Rozmiar danych treningowych:", X_train.shape, y_train.shape)
                st.write("🔍 Typ najlepszego modelu:", type(best_model))
                st.write("🔍 Parametry modelu:", best_model.get_params())

                st.success(f"🏆 Najlepszy model: **{type(best_model).__name__}**, wynik: {best_score:.3f}")

                if hasattr(best_model, "feature_importances_"):
                    fi = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
                    st.bar_chart(fi.head(10))
                    top_features = fi.head(5).index.tolist()
                else:
                    top_features = X.columns[:5].tolist()
                    st.info("ℹ️ Model nie udostępnia informacji o ważności cech.")

                st.subheader("📝 Opis analizy")
                st.markdown(f"""
                **Zadanie:** {problem_type.capitalize()}  
                **Kolumna docelowa:** `{target_col}`  
                **Najlepszy model:** `{type(best_model).__name__}`  
                **Najważniejsze cechy:** {', '.join(top_features)}  
                **Wynik modelu:** {best_score:.3f}

                Model przewiduje wartości na podstawie najistotniejszych cech danych.
                Wynik oznacza {('dokładność klasyfikacji' if problem_type=='classification' else 'siłę dopasowania modelu regresyjnego')}.
                """)
            except Exception as e:
                st.error(f"❌ Błąd podczas trenowania modeli: {e}")
        """)
        except Exception as e:
            st.error(f"❌ Błąd podczas trenowania modeli: {e}")
