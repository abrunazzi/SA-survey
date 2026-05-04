import streamlit as st
import pandas as pd
import re
import nltk
from transformers import pipeline
from keybert import KeyBERT
from collections import Counter
from nltk.corpus import stopwords
import io
from groq import Groq

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Sentiment Analyzer AI", layout="wide")
st.title("🦆 SENTIMENT ANALYSIS & SUMMARIZATION")
st.markdown("Carica un file Excel e seleziona l'analisi desiderata. Quack!")

# --- CARICAMENTO RISORSE (Cache) ---
@st.cache_resource
def load_nltk():
    nltk.download('stopwords')
    return set(stopwords.words('italian'))

@st.cache_resource
def load_models():
    sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", truncation=True)
    kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')
    return sentiment_model, kw_model

stop_words = load_nltk()
sentiment_pipeline, kw_model = load_models()

# Inizializzazione Groq
if "GROQ_API_KEY" in st.secrets:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"].strip())
else:
    st.error("🦆 Chiave API di Groq mancante!")
    st.stop()

# --- FUNZIONI DI ANALISI ---
def run_sentiment(df, column):
    mapping = {'1 star': 'Molto Negativo', '2 stars': 'Negativo', '3 stars': 'Neutro', '4 stars': 'Positivo', '5 stars': 'Molto Positivo'}
    texts = df[column].astype(str).tolist()
    results = sentiment_pipeline(texts, batch_size=8)
    df['Etichetta'] = [mapping.get(res['label'], "Neutro") for res in results]
    df['Confidenza'] = [res['score'] for res in results]
    return df

def run_top_words(df, column):
    keyword_data = []
    for label in df['Etichetta'].unique():
        subset_texts = df[df['Etichetta'] == label][column].astype(str)
        all_words = []
        for text in subset_texts:
            text_clean = re.sub(r'[^a-zA-Zàèìòù\s]', '', text.lower())
            words = [w for w in text_clean.split() if w not in stop_words and len(w) > 2]
            all_words.extend(words)
        most_common = Counter(all_words).most_common(20)
        top_words_string = ", ".join([f"{word} ({count})" for word, count in most_common])
        keyword_data.append({'Sentiment': label, 'Parole più frequenti': top_words_string})
    return pd.DataFrame(keyword_data)

def run_keybert(df, column):
    super_testo = " ".join(df[column].astype(str).tolist())
    keywords = kw_model.extract_keywords(super_testo, keyphrase_ngram_range=(2, 3), stop_words=list(stop_words), top_n=30, use_mmr=True, diversity=0.4)
    return pd.DataFrame(keywords, columns=['Concetto Chiave', 'Rilevanza'])

def genera_riassunto_con_groq(lista_testi, istruzione_utente):
    corpo_testo = "\n- ".join(lista_testi[:300])
    risposta = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Sei un esperto analista di dati. Rispondi in italiano."},
            {"role": "user", "content": f"{istruzione_utente}\n\nDATI:\n{corpo_testo}"}
        ],
        model="llama-3.1-8b-instant",
        temperature=0.3,
    )
    return risposta.choices[0].message.content

# --- INTERFACCIA ---
if "df_processed" not in st.session_state: st.session_state.df_processed = None
if "report_words" not in st.session_state: st.session_state.report_words = None
if "report_keybert" not in st.session_state: st.session_state.report_keybert = None
if "riassunto_ai" not in st.session_state: st.session_state.riassunto_ai = None

uploaded_file = st.file_uploader("🦆 Carica Excel", type=["xlsx"])

if uploaded_file:
    df_input = pd.read_excel(uploaded_file)
    st.write("### 🦆 Anteprima Dati")
    st.dataframe(df_input.head())
    
    colonna_target = st.selectbox("🦆 Seleziona colonna", df_input.columns)
    st.session_state['colonna_target'] = colonna_target

    col1, col2, col3 = st.columns(3)

    if col1.button("🦆 Sentiment Analysis"):
        with st.spinner("Le anatre stanno analizzando..."):
            st.session_state.df_processed = run_sentiment(df_input.copy(), colonna_target)
            st.success("🦆 Analisi completata!")
            st.dataframe(st.session_state.df_processed)

    if col2.button("🦆 Top Words"):
        if st.session_state.df_processed is not None:
            st.session_state.report_words = run_top_words(st.session_state.df_processed, colonna_target)
            st.table(st.session_state.report_words)
        else: st.error("🦆 Esegui prima il Sentiment!")

    if col3.button("🦆 KeyBERT"):
        with st.spinner("Caccia ai concetti in corso..."):
            st.session_state.report_keybert = run_keybert(df_input, colonna_target)
            st.dataframe(st.session_state.report_keybert)

    st.divider()
    st.subheader("🦆 Summarization Intelligente")
    prompt_user = st.text_area("🦆 Istruzione per Riassunto", "Riassumi i punti chiave:")
    if st.button("🦆 Genera Riassunto"):
        testi = df_input[colonna_target].astype(str).tolist()
        with st.spinner("L'anatra robot sta scrivendo..."):
            st.session_state.riassunto_ai = genera_riassunto_con_groq(testi, prompt_user)
            st.info("### 🦆 Report Generato")
            st.markdown(st.session_state.riassunto_ai)

    # --- SEZIONE DOWNLOAD ---
    if st.session_state.df_processed is not None or st.session_state.riassunto_ai is not None:
        st.divider()
        st.subheader("🦆 Area Download Report")
        
        output_excel = io.BytesIO()
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            if st.session_state.df_processed is not None:
                st.session_state.df_processed.to_excel(writer, sheet_name='Sentiment', index=False)
            if st.session_state.report_words is not None:
                st.session_state.report_words.to_excel(writer, sheet_name='Top Words', index=False)
            if st.session_state.report_keybert is not None:
                st.session_state.report_keybert.to_excel(writer, sheet_name='Concetti KeyBERT', index=False)
        
        st.download_button(
            label="🦆 Scarica Report Excel Completo",
            data=output_excel.getvalue(),
            file_name="report_anatra_totale.xlsx",
            mime="application/vnd.ms-excel"
        )

        if st.session_state.riassunto_ai:
            st.download_button(
                label="🦆 Scarica Riassunto AI (.txt)",
                data=st.session_state.riassunto_ai,
                file_name="riassunto_anatra.txt",
                mime="text/plain"
            )
