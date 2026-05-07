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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Sentiment Analyzer AI", layout="wide")
st.title("SENTIMENT ANALYSIS AND SUMMARIZATION")
st.markdown("Carica un file Excel e seleziona l'analisi desiderata.")

# --- CARICAMENTO RISORSE (Cache) ---
@st.cache_resource
def load_nltk():
    nltk.download('stopwords')
    return set(stopwords.words('italian'))

@st.cache_resource
def load_models():
    # Carichiamo BERT per il sentiment
    sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", truncation=True)
    # Carichiamo KeyBERT
    kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')
    # Carichiamo il modello per gli Embedding (base di BERTopic)
    embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return sentiment_model, kw_model, embed_model

stop_words = load_nltk()
sentiment_pipeline, kw_model, embed_model = load_models()

# Inizializzazione Groq
if "GROQ_API_KEY" in st.secrets:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"].strip())
else:
    st.error("Chiave API di Groq mancante nei Secrets.")
    st.stop()

# --- FUNZIONI DI ANALISI ---

def pulisci_lista_testi(df, column):
    testi_puliti = df[column].dropna().astype(str).tolist()
    testi_puliti = [t.strip() for t in testi_puliti if len(t.strip()) > 0]
    return testi_puliti

def run_sentiment(df, column):
    mapping = {'1 star': 'Molto Negativo', '2 stars': 'Negativo', '3 stars': 'Neutro', '4 stars': 'Positivo', '5 stars': 'Molto Positivo'}
    df_clean = df.copy()
    df_clean[column] = df_clean[column].fillna("Dato non disponibile").astype(str)
    texts = df_clean[column].tolist()
    results = sentiment_pipeline(texts, batch_size=8)
    df_clean['Etichetta'] = [mapping.get(res['label'], "Neutro") for res in results]
    df_clean['Confidenza'] = [res['score'] for res in results]
    return df_clean

def run_top_words(df, column):
    keyword_data = []
    if 'Etichetta' not in df.columns:
        return pd.DataFrame()
    for label in df['Etichetta'].unique():
        subset_texts = df[df['Etichetta'] == label][column].astype(str)
        all_words = []
        for text in subset_texts:
            text_clean = re.sub(r'[^a-zA-Zàèìòù\s]', '', text.lower())
            words = [w for w in text_clean.split() if w not in stop_words and len(w) > 2]
            all_words.extend(words)
        most_common = Counter(all_words).most_common(20)
        top_words_string = ", ".join([f"{word} ({count})" for word, count in most_common])
        keyword_data.append({'Sentiment': label, 'Parole piu frequenti': top_words_string})
    return pd.DataFrame(keyword_data)

def run_keybert(df, column):
    testi_per_keybert = pulisci_lista_testi(df, column)
    super_testo = " ".join(testi_per_keybert)
    if not super_testo:
        return pd.DataFrame(columns=['Concetto Chiave', 'Rilevanza'])
    keywords = kw_model.extract_keywords(
        super_testo, 
        keyphrase_ngram_range=(2, 3), 
        stop_words=list(stop_words), 
        top_n=30, 
        use_mmr=True, 
        diversity=0.4
    )
    return pd.DataFrame(keywords, columns=['Concetto Chiave', 'Rilevanza'])

def run_topic_modeling(df, column, n_topics=5):
    testi = pulisci_lista_testi(df, column)
    if len(testi) < 10:
        return pd.DataFrame([{"Avviso": "Dati insufficienti"}])
    vectorizer = CountVectorizer(stop_words=list(stop_words), max_features=10000)
    data_vectorized = vectorizer.fit_transform(testi)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_output = lda.fit_transform(data_vectorized)
    topic_assignments = lda_output.argmax(axis=1)
    counts = Counter(topic_assignments)
    words = vectorizer.get_feature_names_out()
    topic_data = []
    for i, topic in enumerate(lda.components_):
        top_words = [words[j] for j in topic.argsort()[-10:]]
        topic_data.append({
            "Macro-Tema": f"Gruppo {i+1}",
            "Conteggio Frasi": counts.get(i, 0),
            "Parole Chiave": ", ".join(reversed(top_words))
        })
    return pd.DataFrame(topic_data).sort_values(by="Conteggio Frasi", ascending=False)

def run_semantic_clustering(df, column, n_clusters=5):
    """Implementazione stile BERTopic: Embedding + Clustering + LLM Labeling"""
    testi = pulisci_lista_testi(df, column)
    if len(testi) < n_clusters:
        return pd.DataFrame([{"Avviso": "Dati insufficienti per il clustering semantico"}])
    
    # 1. Creazione degli Embedding (Vettori)
    embeddings = embed_model.encode(testi)
    
    # 2. Clustering con K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    df_temp = pd.DataFrame({"testo": testi, "cluster": cluster_labels})
    counts = Counter(cluster_labels)
    
    topic_data = []
    for i in range(n_clusters):
        # Prendiamo 10 esempi reali per farli leggere a Llama
        campioni = df_temp[df_temp['cluster'] == i]['testo'].head(10).tolist()
        corpo_campioni = "\n- ".join(campioni)
        
        # 3. Chiediamo a Llama di dare un nome al tema basandosi sul significato
        risposta = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Sei un analista. Guarda gli esempi e scrivi solo un titolo di 3-4 parole che riassuma il tema."},
                {"role": "user", "content": f"Esempi di risposte:\n{corpo_campioni}"}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.1,
        )
        nome_tema = risposta.choices[0].message.content.strip()
        
        topic_data.append({
            "Tema Semantico (BERTopic style)": nome_tema,
            "Conteggio": counts.get(i, 0),
            "Esempio reale": campioni[0]
        })
    
    return pd.DataFrame(topic_data).sort_values(by="Conteggio", ascending=False)

def genera_riassunto_con_groq(lista_testi, istruzione_utente):
    corpo_testo = "\n- ".join(lista_testi[:300])
    risposta = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Sei un analista di survey esperto. Rispondi in italiano."},
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
if "report_topics" not in st.session_state: st.session_state.report_topics = None
if "report_semantic" not in st.session_state: st.session_state.report_semantic = None
if "riassunto_ai" not in st.session_state: st.session_state.riassunto_ai = None

uploaded_file = st.file_uploader("Carica file Excel", type=["xlsx"])

if uploaded_file:
    df_input = pd.read_excel(uploaded_file)
    st.write("### Anteprima Dati")
    st.dataframe(df_input.head())
    
    colonna_target = st.selectbox("Seleziona colonna testi", df_input.columns)
    st.session_state['colonna_target'] = colonna_target

    col1, col2, col3, col4, col5 = st.columns(5)

    if col1.button("Sentiment"):
        with st.spinner("Analisi..."):
            try:
                st.session_state.df_processed = run_sentiment(df_input, colonna_target)
                st.dataframe(st.session_state.df_processed)
            except Exception as e: st.error(f"Errore: {e}")

    if col2.button("Top Words"):
        if st.session_state.df_processed is not None:
            st.session_state.report_words = run_top_words(st.session_state.df_processed, colonna_target)
            st.table(st.session_state.report_words)
        else: st.error("Esegui Sentiment prima.")

    if col3.button("KeyBERT"):
        with st.spinner("Estrazione..."):
            st.session_state.report_keybert = run_keybert(df_input, colonna_target)
            st.dataframe(st.session_state.report_keybert)

    if col4.button("Temi LDA"):
        with st.spinner("LDA in corso..."):
            st.session_state.report_topics = run_topic_modeling(df_input, colonna_target)
            st.table(st.session_state.report_topics)

    if col5.button("Temi BERT"):
        with st.spinner("Analisi Semantica (BERTopic style)..."):
            try:
                st.session_state.report_semantic = run_semantic_clustering(df_input, colonna_target)
                st.table(st.session_state.report_semantic)
            except Exception as e: st.error(f"Errore Clustering: {e}")

    st.divider()
    st.subheader("Summarization Intelligente")
    prompt_user = st.text_area("Istruzione per il riassunto", "Analizza queste risposte e riassumi i punti chiave:")
    
    if st.button("Genera Riassunto"):
        testi_puliti = pulisci_lista_testi(df_input, colonna_target)
        if testi_puliti:
            with st.spinner("Generazione report..."):
                try:
                    st.session_state.riassunto_ai = genera_riassunto_con_groq(testi_puliti, prompt_user)
                    st.markdown(st.session_state.riassunto_ai)
                except Exception as e: st.error(f"Errore Groq: {e}")

    # --- SEZIONE DOWNLOAD ---
    if st.session_state.df_processed is not None or st.session_state.riassunto_ai is not None:
        st.divider()
        st.subheader("Area Download")
        output_excel = io.BytesIO()
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            if st.session_state.df_processed is not None:
                st.session_state.df_processed.to_excel(writer, sheet_name='Sentiment', index=False)
            if st.session_state.report_words is not None:
                st.session_state.report_words.to_excel(writer, sheet_name='Top Words', index=False)
            if st.session_state.report_keybert is not None:
                st.session_state.report_keybert.to_excel(writer, sheet_name='KeyBERT', index=False)
            if st.session_state.report_topics is not None:
                st.session_state.report_topics.to_excel(writer, sheet_name='Temi LDA', index=False)
            if st.session_state.report_semantic is not None:
                st.session_state.report_semantic.to_excel(writer, sheet_name='Temi Semantici BERT', index=False)
        
        st.download_button(label="Scarica Excel Completo", data=output_excel.getvalue(), file_name="report.xlsx")
