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

import streamlit as st
import pandas as pd
from llama_cpp import Llama
import io

# --- CARICAMENTO MODELLO LLM (Cache per non ricaricarlo ogni volta) ---
@st.cache_resource
def load_llm():
    # Modifica il percorso con quello dove tieni il file .gguf sul tuo PC
    # Se lo carichi su GitHub, dovrai usare un server con GPU
    model_path = "MODELLI_AI/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    return Llama(
        model_path=model_path,
        n_gpu_layers=-1, # Usa -1 se hai una GPU NVIDIA, altrimenti 0
        n_ctx=4096
    )

# --- INTERFACCIA STREAMLIT ---
st.header("🤖 Summarization Intelligente")

# 1. Casella per personalizzare il Prompt
# Usiamo i tag [INST] e [/INST] suggeriti dal modello Mistral
default_prompt = "Le seguenti sono risposte a un sondaggio. Analizzale e riassumi in italiano cosa piace e cosa non piace agli utenti:"
user_instruction = st.text_area("Inserisci l'istruzione per l'IA (all'interno di [INST]):", value=default_prompt)

# 2. Bottone per avviare il riassunto
if st.button("📝 Genera Report Testuale"):
    if st.session_state.df_processed is not None: # Controlla se hai già caricato i dati
        with st.spinner("L'IA sta leggendo e riassumendo... attendi..."):
            
            # Recuperiamo il testo dalla colonna target
            # (Assumiamo che tu abbia salvato la colonna scelta nello stato)
            testi = st.session_state.df_processed[st.session_state.colonna_scelta].dropna().astype(str).tolist()
            testo_unito = "\n- ".join(testi[:200]) # Limite a 200 per non saturare la memoria

            # Costruzione del Prompt Finale
            full_prompt = f"<s>[INST] {user_instruction} \n{testo_unito} [/INST]</s>"

            # Caricamento ed esecuzione
            llm = load_llm()
            output = llm(
                full_prompt,
                max_tokens=1000,
                temperature=0.4,
                echo=False
            )
            
            riassunto = output['choices'][0]['text']

            # Mostra il risultato
            st.markdown("### 📄 Analisi Generata")
            st.write(riassunto)

            # Bottone per scaricare il report in TXT
            st.download_button(
                label="📥 Scarica Report (.txt)",
                data=riassunto,
                file_name="report_ai.txt",
                mime="text/plain"
            )
    else:
        st.error("Per favore, carica prima un file e avvia la Sentiment Analysis!")
