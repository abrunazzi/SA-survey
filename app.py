import streamlit as st
import pandas as pd
import re
import nltk
from transformers import pipeline
from keybert import KeyBERT
from collections import Counter
from nltk.corpus import stopwords
import io

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title=" BERT", layout="wide")
st.title("SENTIMENT ANALYSIS")
st.markdown("Carica un file Excel e seleziona l'analisi desiderata.")

# --- DOWNLOAD RISORSE NLTK ---
@st.cache_resource
def load_nltk():
    nltk.download('stopwords')
    return set(stopwords.words('italian'))

stop_words = load_nltk()

# --- CARICAMENTO MODELLI (In Cache) ---
@st.cache_resource
def load_models():
    sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", truncation=True)
    kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')
    return sentiment_model, kw_model

sentiment_pipeline, kw_model = load_models()

# --- FUNZIONI DI ANALISI ---
def run_sentiment(df, column):
    mapping = {
        '1 star': 'Molto Negativo', '2 stars': 'Negativo',
        '3 stars': 'Neutro', '4 stars': 'Positivo', '5 stars': 'Molto Positivo'
    }
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
    keywords = kw_model.extract_keywords(
        super_testo, keyphrase_ngram_range=(2, 3), 
        stop_words=list(stop_words), top_n=30, use_mmr=True, diversity=0.4
    )
    return pd.DataFrame(keywords, columns=['Concetto Chiave', 'Rilevanza'])

# --- INTERFACCIA DRAG & DROP ---
uploaded_file = st.file_uploader("Trascina qui il tuo file Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    df_input = pd.read_excel(uploaded_file)
    st.write("### Anteprima Dati")
    st.dataframe(df_input.head())

    colonna_target = st.selectbox("Seleziona la colonna con i testi da analizzare", df_input.columns)

    # --- BOTTONI AZIONE ---
    st.write("---")
    col1, col2, col3 = st.columns(3)

    if "df_processed" not in st.session_state:
        st.session_state.df_processed = None

    if col1.button(" Avvia Sentiment Analysis"):
        with st.spinner("Analisi in corso..."):
            st.session_state.df_processed = run_sentiment(df_input, colonna_target)
            st.success("Analisi completata!")
            st.dataframe(st.session_state.df_processed)

    if col2.button(" Genera Top Words"):
        if st.session_state.df_processed is not None:
            report_words = run_top_words(st.session_state.df_processed, colonna_target)
            st.write("### Report Parole per Categoria")
            st.table(report_words)
        else:
            st.error("Esegui prima la Sentiment Analysis!")

    if col3.button("Estrai Concetti (KeyBERT)"):
        with st.spinner("Estrazione semantica..."):
            report_keybert = run_keybert(df_input, colonna_target)
            st.write("### Concetti Chiave Rilevati")
            st.dataframe(report_keybert)

    # --- DOWNLOAD DEI RISULTATI ---
    if st.session_state.df_processed is not None:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            st.session_state.df_processed.to_excel(writer, index=False)
        st.download_button(
            label=" Scarica Risultati Sentiment",
            data=output.getvalue(),
            file_name="analisi_sentiment.xlsx",
            mime="application/vnd.ms-excel"
        )

from groq import Groq

# Inizializza il client usando la chiave salvata nei Secrets
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def genera_riassunto_con_groq(lista_testi, istruzione_utente):
    # Prepariamo il testo unito (prendiamo le prime 300 risposte per stare sicuri)
    corpo_testo = "\n- ".join(lista_testi[:300])
    
    # Chiamata all'API di Groq
    risposta = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Sei un esperto analista di dati. Rispondi sempre in italiano."
            },
            {
                "role": "user",
                "content": f"{istruzione_utente}\n\nDATI:\n{corpo_testo}"
            }
        ],
        model="llama3-8b-8192", # Modello Llama 3 velocissimo
        temperature=0.3,
    )
    return risposta.choices[0].message.content

# --- Sezione Interfaccia ---
st.divider()
st.subheader(" Summarization ")
prompt_personalizzato = st.text_area("Cosa vuoi chiedere all'IA?", 
                                    "Analizza queste risposte e riassumi i punti chiave su cosa piace e cosa no:")

if st.button(" Genera Riassunto con Llama 3"):
    if st.session_state.df_processed is not None:
        with st.spinner("L'IA sta leggendo le risposte..."):
            # Recuperiamo i testi originali
            testi_da_analizzare = st.session_state.df_processed[st.session_state.colonna_target].astype(str).tolist()
            
            # Eseguiamo la funzione
            risultato_ai = genera_riassunto_con_groq(testi_da_analizzare, prompt_personalizzato)
            
            # Mostriamo il risultato
            st.info("### Report Generato")
            st.markdown(risultato_ai)
    else:
        st.warning("Carica prima il file ed esegui la Sentiment Analysis!")
