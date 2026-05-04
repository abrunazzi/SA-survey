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
st.markdown("Carica un file Excel e seleziona l'analisi desiderata. La SA è obbligatoria per gli step successivi. ")

# --- DOWNLOAD RISORSE NLTK ---
@st.cache_resource
def load_nltk():
    nltk.download('stopwords')
    return set(stopwords.words('italian'))

stop_words = load_nltk()

# --- CARICAMENTO MODELLI (In Cache) ---
@st.cache_resource
def load_models():
    # Carichiamo BERT per il sentiment
    sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", truncation=True)
    # Carichiamo KeyBERT per i concetti
    kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')
    return sentiment_model, kw_model

sentiment_pipeline, kw_model = load_models()

# --- INIZIALIZZAZIONE CLIENT GROQ ---
if "GROQ_API_KEY" in st.secrets:
    # .strip() rimuove eventuali spazi bianchi o invii accidentali nella chiave
    api_key_val = st.secrets["GROQ_API_KEY"].strip()
    client = Groq(api_key=api_key_val)
else:
    st.error("⚠️ Chiave API di Groq non trovata nei Secrets! L'analisi testuale non funzionerà.")
    st.stop()

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
        most_common = Counter(all_words).most_common(40)
        top_words_string = ", ".join([f"{word} ({count})" for word, count in most_common])
        keyword_data.append({'Sentiment': label, 'Parole più frequenti': top_words_string})
    return pd.DataFrame(keyword_data)

def run_keybert(df, column):
    super_testo = " ".join(df[column].astype(str).tolist())
    keywords = kw_model.extract_keywords(
        super_testo, keyphrase_ngram_range=(2, 3), 
        stop_words=list(stop_words), top_n=40, use_mmr=True, diversity=0.3
    )
    return pd.DataFrame(keywords, columns=['Concetto Chiave', 'Rilevanza'])

def genera_riassunto_con_groq(lista_testi, istruzione_utente):
    corpo_testo = "\n- ".join(lista_testi[:300]) # Limite per i token di input
    
    # Utilizzo di llama-3.1-8b-instant per evitare l'errore di decommissioning
    risposta = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Queste sono le risposte aperte di 200 persone diverse alla domanda ......"},
            {"role": "user", "content": f"{istruzione_utente}\n\nDATI:\n{corpo_testo}"}
        ],
        model="llama-3.1-8b-instant",
        temperature=0.4,
    )
    return risposta.choices[0].message.content

# --- INTERFACCIA PRINCIPALE ---
uploaded_file = st.file_uploader("Trascina qui il tuo file Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    df_input = pd.read_excel(uploaded_file)
    st.write("### Anteprima Dati")
    st.dataframe(df_input.head())

    # Salviamo la colonna target in session_state per garantirne la persistenza
    colonna_target = st.selectbox("Seleziona la colonna con i testi da analizzare 🦆", df_input.columns)
    st.session_state['colonna_target'] = colonna_target

    st.write("---")
    col1, col2, col3 = st.columns(3)

    if "df_processed" not in st.session_state:
        st.session_state.df_processed = None

    # BOTTONE 1: SENTIMENT
    if col1.button(" Sentiment Analysis"):
        with st.spinner("Analisi BERT in corso..."):
            st.session_state.df_processed = run_sentiment(df_input.copy(), colonna_target)
            st.success("Analisi completata!")
            st.dataframe(st.session_state.df_processed)

    # BOTTONE 2: TOP WORDS
    if col2.button(" Top Words"):
        if st.session_state.df_processed is not None:
            target = st.session_state['colonna_target']
            report_words = run_top_words(st.session_state.df_processed, target)
            st.write("### Report Parole per Categoria")
            st.table(report_words)
        else:
            st.error("Esegui prima la Sentiment Analysis!")

    # BOTTONE 3: KEYBERT
    if col3.button(" Concetti (KeyBERT)"):
        with st.spinner("Estrazione semantica..."):
            target = st.session_state['colonna_target']
            report_keybert = run_keybert(df_input, target)
            st.write("### Concetti Chiave Rilevati")
            st.dataframe(report_keybert)

    # --- SEZIONE SUMMARIZATION (GROQ) ---
    st.divider()
    st.subheader("🦆 Summarization Intelligente")
    prompt_personalizzato = st.text_area("Cosa vuoi chiedere all'IA?", 
                                        "Analizza queste risposte e riassumi i punti chiave su cosa piace e cosa no:")

    if st.button(" Genera Riassunto con Llama 3.1"):
        # Controllo sicurezza session_state
        if 'colonna_target' in st.session_state and st.session_state['colonna_target'] in df_input.columns:
            with st.spinner("Groq sta elaborando..."):
                try:
                    target = st.session_state['colonna_target']
                    testi_da_analizzare = df_input[target].astype(str).tolist()
                    risultato_ai = genera_riassunto_con_groq(testi_da_analizzare, prompt_personalizzato)
                    st.info("### Report Generato")
                    st.markdown(risultato_ai)
                except Exception as e:
                    st.error(f"Errore durante la generazione: {e}")
        else:
            st.warning("Assicurati di aver selezionato una colonna valida prima di generare il riassunto.")

    # --- DOWNLOAD DEI RISULTATI ---
    if st.session_state.df_processed is not None:
        st.write("---")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            st.session_state.df_processed.to_excel(writer, index=False)
        st.download_button(
            label="🦆 Scarica Risultati Sentiment (Excel)",
            data=output.getvalue(),
            file_name="analisi_sentiment_completa.xlsx",
            mime="application/vnd.ms-excel"
        )
