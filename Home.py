# Import library yang diperlukan
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
import pandas as pd
import math
import numpy as np

nltk.download("stopwords")

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Information Retrieval", layout="wide")

# Menampilkan judul aplikasi
st.title("Welcome to Information Retrieval System! 👋")
st.write("Aplikasi Information Retrieval ini memungkinkan pengguna untuk melakukan pencarian dokumen menggunakan metode-metode yang terkenal dalam bidang ini. Dengan adanya metode Boolean Model, TF-IDF, dan Vector Space Model, pengguna dapat dengan mudah mencari dokumen yang relevan dengan query yang diberikan")

# Fungsi untuk menghapus karakter khusus dari teks
def remove_special_characters(text):
    regex = re.compile('[^a-zA-Z0-9\s]')
    text_returned = re.sub(regex, '', text)
    return text_returned

# Fungsi untuk melakukan pra-pemrosesan pada teks
def preprocess(text):
    text = remove_special_characters(text)
    text = re.sub(re.compile('\d'), '', text)
    words = word_tokenize(text)
    if use_stem_or_lem == "Stemming":
        if stopword_lang == "Bahasa":
            stemmer = StemmerFactory().create_stemmer()
        else:
            stemmer = nltk.stem.PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    elif use_stem_or_lem == "Lemmatization":
        lemmatizer = nltk.stem.WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
    if is_using_stopword:
        words = [word.lower() for word in words if word not in Stopwords]
    else:
        words = [word.lower() for word in words]
    return words

# Fungsi untuk mencari kata-kata unik dan frekuensinya
def finding_all_unique_words_and_freq(words):
    word_freq = {}
    for word in words:
        word_freq[word] = words.count(word)
    return word_freq

# Fungsi untuk membangun indeks dan file terindeks
def build_index(text_list):
    idx = 1
    indexed_files = {}
    index = {}
    for text in text_list:
        words = preprocess(text)
        indexed_files[idx] = f"dokumen{idx}"
        for word, freq in finding_all_unique_words_and_freq(words).items():
            if word not in index:
                index[word] = {}
            index[word][idx] = freq
        idx += 1
    return index, indexed_files

# Fungsi untuk membangun tabel dari data
def build_table(data):
    rows = []
    for key, val in data.items():
        row = [key, val]
        rows.append(row)
    return rows

# Fungsi untuk membangun tabel matriks kejadian
def build_table_incidence_matrix(data, indexed_files):
    rows = []
    for key, val in data.items():
        row = [key]
        for file_id, file_name in indexed_files.items():
            if file_id in val:
                row.append("1")
            else:
                row.append("0")
        rows.append(row)
    return rows

# Fungsi untuk melakukan pencarian
def search(query_words, index, indexed_files):
    connecting_words = []
    different_words = []
    for word in query_words:
        if word.lower() in ["and", "or", "not"]:
            connecting_words.append(word.lower())
        else:
            different_words.append(word.lower())
    if not different_words:
        st.write("Harap masukkan kata kunci query")
        return []
    results = set(index[different_words[0]])
    for word in different_words[1:]:
        if word.lower() in index:
            results = set(index[word.lower()]) & results
        else:
            st.write(f"{word} tidak ditemukan dalam dokumen")
            return []
    for word in connecting_words:
        if word == "and":
            next_results = set(index[different_words[0]])
            for word in different_words[1:]:
                if word.lower() in index:
                    next_results = set(index[word.lower()]) & next_results
                else:
                    st.write(f"{word} tidak ditemukan dalam dokumen")
                    return []
            results = results & next_results
        elif word == "or":
            next_results = set(index[different_words[0]])
            for word in different_words[1:]:
                if word.lower() in index:
                    next_results = set(index[word.lower()]) | next_results
            results = results | next_results
        elif word == "not":
            not_results = set()
            for word in different_words[1:]:
                if word.lower() in index:
                    not_results = not_results | set(index[word.lower()])
            results = set(index[different_words[0]]) - not_results
    return results

def boolean_search(query_words, index, indexed_files):
    results = set()
    
    for word in query_words:
        if word.lower() in index:
            files = index[word.lower()]
            results.update(files)
    
    return results

def merge_results(results1, results2, operator):
    if operator == 'AND':
        return results1.intersection(results2)
    elif operator == 'OR':
        return results1.union(results2)
    elif operator == 'NOT':
        return results1.difference(results2)
    else:
        raise ValueError("Operator not supported.")

tab1, tab2, tab0= st.tabs(
    ["Dokumen", "Pre-processing","Query"])

# Tab 1: Dokumen
with tab1:
    st.subheader("Dokumen")
    uploaded_files = st.file_uploader(
        "Upload one or more files", accept_multiple_files=True,  key='tab1')

    text_list = []
    for file in uploaded_files:
        file_type = file.name.split('.')[-1]
        if file_type == 'txt':
            text_list.append(file.read().decode('utf-8'))
        elif file_type == 'csv':
            df = pd.read_csv(file)
            text_list.extend(df.iloc[:, 0].dropna().tolist())

    D = len(text_list)

    # menampilkan preview dokumen yang diupload
    if text_list:
        st.subheader("Isi Dokumen")

        df_dokumen = pd.DataFrame({
            'Dokumen': ['Dokumen '+str(i+1) for i in range(D)],
            'Isi': text_list
        })
        st.table(df_dokumen)

# Tab 2: Pre-Processing
with tab2:
    st.subheader("Pengaturan Preprocessing")
    use_stem_or_lem = st.selectbox(
        "Stemming/Lemmatization", ("Stemming", "Lemmatization"))
    is_using_stopword = st.checkbox("Stopword Removal", value=True)
    stopword_lang = st.selectbox("Stopwords Language", ("Bahasa", "English"))

    if stopword_lang == "Bahasa":
        Stopwords = set(open('stopwordid.txt').read().split())
    else:
        Stopwords = set(stopwords.words('english'))

    documents = []
    for text in text_list:
        documents.append(preprocess(text))

    # tokenisasi
    tokens = [doc for doc in documents]

    st.subheader("Preprocessing Pada Tiap Dokumen:")
    df_token = pd.DataFrame({
        'Dokumen': ['Dokumen '+str(i+1) for i in range(D)],
        'Token': tokens
    })
    st.table(df_token)

with tab0:
    query = st.text_input('Masukkan kata kunci query:')
    query_words = word_tokenize(query)
    query = preprocess(query)

# Memeriksa apakah pengguna telah memasukkan query
if query:
    # Menampilkan tab-tab setelah pengguna memasukkan query
    tab3, tab4, tab5 = st.tabs(
        ["Boolean Model", "TF-IDF", "Vector Space Model"])

    # Tab 3 : Boolean Model
    with tab3:
        st.subheader("Boolean Model")
        st.write("Information Retrieval System dengan model Boolean menggunakan incidence matrix dan inverted index adalah sistem yang memungkinkan pengguna untuk mencari dan mengambil dokumen yang relevan berdasarkan pertanyaan atau kueri yang mengandung operator boolean. Sistem ini menggunakan incidence matrix untuk merepresentasikan kehadiran atau ketidakhadiran kata kunci dalam setiap dokumen, sementara inverted index digunakan untuk mengindeks kata kunci dan menunjukkan dokumen mana yang mengandung kata kunci tersebut")

        st.markdown("""
        <style>
        table td:nth-child(1) {
            display: none
        }
        table th:nth-child(1) {
            display: none
        }
        </style>
        """, unsafe_allow_html=True)

        index, indexed_files = build_index(text_list)

        if query_words:
            inverted_index_table = build_table(index)

            # Mendapatkan hasil pencarian menggunakan operasi AND, OR, atau NOT
            results = set()
            operator = None
            for word in query_words:
                if word.lower() == 'and' or word.lower() == 'or' or word.lower() == 'not':
                    operator = word.upper()
                else:
                    if operator is None:
                        results = boolean_search([word], index, indexed_files)
                    else:
                        results = merge_results(results, boolean_search([word], index, indexed_files), operator)
                    operator = None
            
            results_files = [indexed_files[file_id] for file_id in results]

            st.subheader("Inverted Index")
            df_inverted_index_table = pd.DataFrame(
                inverted_index_table, columns=["Term", "Posting List"])
            st.table(df_inverted_index_table)

            st.subheader("Incidence Matrix")
            incidence_matrix_table_header = [
                "Term"] + [file_name for file_name in indexed_files.values()]
            incidence_matrix_table = build_table_incidence_matrix(
                index, indexed_files)
            df_incidence_matrix_table = pd.DataFrame(
                incidence_matrix_table, columns=incidence_matrix_table_header)
            st.table(df_incidence_matrix_table)

            if not results_files:
                st.warning("Kata kunci tidak ditemukan dalam dokumen")
            else:
                st.subheader("Hasil")
                st.markdown(f"""
                        Dokumen yang relevan dengan query adalah:
                            **{', '.join(results_files)}**
                        """)

    # Tab 4 : TF-IDF
    with tab4:
        st.subheader("TF-IDF")
        st.write("TF-IDF (Term Frequency-Inverse Document Frequency) adalah sebuah metode statistik yang digunakan untuk mengukur pentingnya sebuah kata (term) dalam sebuah dokumen atau koleksi dokumen. Metode ini umum digunakan dalam pengambilan informasi dan penambangan teks untuk menentukan relevansi suatu dokumen terhadap sebuah query.")

        # Mengubah tampilan tabel dengan CSS
        st.markdown("""
        <style>
        table td:nth-child(1) {
            display: none
        }
        table th:nth-child(1) {
            display: none
        }
        </style>
        """, unsafe_allow_html=True)

        # menghitung df dan menghitung idf
        df = {}
        D = len(documents)
        for i in range(D):
            for token in set(tokens[i]):
                if token not in df:
                    df[token] = 1
                else:
                    df[token] += 1

        idf = {token: math.log10(D/df[token]) for token in df}

        # menghitung tf
        tf = []
        for i in range(D):
            tf.append({})
            for token in tokens[i]:
                if token not in tf[i]:
                    tf[i][token] = 1
                else:
                    tf[i][token] += 1

        # menghitung bobot tf-idf
        tfidf = []
        for i in range(D):
            tfidf.append({})
            for token in tf[i]:
                tfidf[i][token] = tf[i][token] * idf[token]

        # menyimpan hasil pada dataframe
        df_result = pd.DataFrame(columns=['Q'] + ['tf_d'+str(i+1) for i in range(D)] + [
            'df', 'D/df', 'IDF', 'IDF+1'] + ['weight_d'+str(i+1) for i in range(D)])
        for token in query:
            row = {'Q': token}
            for i in range(D):
                # tf_i
                if token in tf[i]:
                    row['tf_d'+str(i+1)] = tf[i][token]
                else:
                    row['tf_d'+str(i+1)] = 0
                # weight_i
                if token in tfidf[i]:
                    row['weight_d'+str(i+1)] = tfidf[i][token] + 1
                else:
                    row['weight_d'+str(i+1)] = 0
            # df
            if token in df:
                df_ = df[token]
            else:
                df_ = 0

            # D/df
            if df_ > 0:
                D_df = D / df_
            else:
                D_df = 0

            # IDF
            if token in idf:
                IDF = idf[token]
            else:
                IDF = 0

            # IDF+1
            IDF_1 = IDF + 1

            row['df'] = df_
            row['D/df'] = D_df
            row['IDF'] = IDF
            row['IDF+1'] = IDF_1

            df_result = df_result.append(row, ignore_index=True)

        # menampilkan output pada Streamlit
        if query:
            st.subheader("Hasil")
            st.subheader("Preprocessing Query:")
            df_query = pd.DataFrame({
                'Query': [query]
            })
            st.table(df_query)
            st.subheader("Tabel TF-IDF")
            st.table(df_result)

            st.subheader("Rangking Dokumen:")
            df_weight_sorted = pd.DataFrame({
                'Dokumen': ['Dokumen '+str(i+1) for i in range(D)],
                'Sum Weight': [sum([df_result['weight_d'+str(i+1)][j] for j in range(len(df_result))]) for i in range(D)]
            })
            st.table(df_weight_sorted.sort_values(
                by=['Sum Weight'], ascending=False))

    # Tab 5 : Vector Space Model
    with tab5:
        st.subheader("Vector Space Model")
        st.write("Vector Space Model (VSM) adalah sebuah model matematis yang digunakan dalam pengambilan informasi dan penambangan teks untuk mewakili dokumen dan pertanyaan sebagai vektor dalam ruang berdimensi tinggi. Dalam model ini, dokumen dan pertanyaan direpresentasikan sebagai vektor di mana setiap dimensi mewakili kata-kata unik. Kemiripan antara dokumen dan pertanyaan diukur dengan menghitung jarak kosinus antara vektor-vektor tersebut. VSM memungkinkan pencarian dokumen yang relevan dengan membandingkan kesamaan vektor antara dokumen dan pertanyaan, dan digunakan dalam berbagai aplikasi seperti mesin pencari dan sistem pengambilan informasi.")

        # tokenisasi
        tokens = [query] + [doc for doc in documents]
        lexicon = []
        for token in tokens:
            for word in token:
                if word not in lexicon:
                    lexicon.append(word)

        # menghitung df dan menghitung idf
        df = {}
        D = len(documents) + 1
        for i in range(D):
            for token in set(tokens[i]):
                if token not in df:
                    df[token] = 1
                else:
                    df[token] += 1

        idf = {token: math.log10(D/df[token]) for token in df}

        # menghitung tf
        tf = []
        for i in range(D):
            tf.append({})
            for token in tokens[i]:
                if token not in tf[i]:
                    tf[i][token] = 1
                else:
                    tf[i][token] += 1

        # menghitung bobot tf-idf
        tfidf = []
        for i in range(D):
            tfidf.append({})
            for token in tf[i]:
                tfidf[i][token] = tf[i][token] * idf[token]

        # menyimpan hasil pada dataframe
        df_result = pd.DataFrame(columns=['token'] + ['tf_Q'] + ['tf_d'+str(i) for i in range(1, D)] + [
            'df', 'D/df', 'IDF', 'IDF+1'] + ['weight_Q'] + ['weight_d'+str(i) for i in range(1, D)])

        for token in lexicon:
            row = {'token': token}
            if token in tf[0]:
                row['tf_Q'] = tf[0][token]
            else:
                row['tf_Q'] = 0

            if token in tfidf[0]:
                row['weight_Q'] = tfidf[0][token]
            else:
                row['weight_Q'] = 0

            for i in range(1, D):
                # tf_i
                if token in tf[i]:
                    row['tf_d'+str(i)] = tf[i][token]
                else:
                    row['tf_d'+str(i)] = 0
                # weight_i
                if token in tfidf[i]:
                    row['weight_d'+str(i)] = tfidf[i][token] + 1
                else:
                    row['weight_d'+str(i)] = 0
            # df
            if token in df:
                df_ = df[token]
            else:
                df_ = 0

            # D/df
            if df_ > 0:
                D_df = D / df_
            else:
                D_df = 0

            # IDF
            if token in idf:
                IDF = idf[token]
            else:
                IDF = 0

            # IDF+1
            IDF_1 = IDF + 1
            row['df'] = df_
            row['D/df'] = D_df
            row['IDF'] = IDF
            row['IDF+1'] = IDF_1

            df_result = pd.concat(
                [df_result, pd.DataFrame(row, index=[0])], ignore_index=True)

        # menampilkan output pada Streamlit
        if query:
            st.subheader("Result")
            st.subheader("Preprocessing Query:")
            df_query = pd.DataFrame({
                'Query': [query]
            })
            st.table(df_query.round(2))

            st.subheader("TF-IDF Table query")
            st.table(df_result)

            st.subheader("Hasil perhitungan jarak Dokumen dengan Query")
            df_distance = pd.DataFrame(
                columns=['Token'] + ['Q' + chr(178)] + ['D'+str(i) + chr(178) for i in range(1, D)])
            df_distance['Token'] = lexicon
            df_distance['Q' + chr(178)] = df_result['weight_Q'] ** 2
            for i in range(1, D):
                df_distance['D'+str(i) + chr(178)
                            ] = df_result['weight_d'+str(i)] ** 2
            st.table(df_distance)
            sqrt_q = round(math.sqrt(df_distance['Q' + chr(178)].sum()), 4)
            sqrt_d = []
            for i in range(1, D):
                sqrt_d.append(
                    round(math.sqrt(df_distance['D'+str(i) + chr(178)].sum()), 4))
                
            st.latex(r'''Sqrt(D_i)= \sqrt{(\sum D_i^2)} ''')

            for i in range(1, D):
                st.latex(
                    r'''Sqrt(D''' + str(i) + r''')= \sqrt{(''' + '+'.join(
                        [str(round(key, 4)) for key in list(df_distance['D' + str(i) + chr(178)])]) + ''')}= ''' + str(sqrt_d[i-1]) + r''' '''
                )

            st.subheader("Perhitungan Vector Space Model")
            df_space_vector = pd.DataFrame(
                columns=['Token'] + ['Q' + chr(178)] + ['D'+str(i) + chr(178) for i in range(1, D)] + ['Q*D'+str(i) for i in range(1, D)])
            df_space_vector['Token'] = lexicon
            df_space_vector['Q' + chr(178)] = df_result['weight_Q'] ** 2
            for i in range(1, D):
                df_space_vector['D'+str(i) + chr(178)
                                ] = df_result['weight_d'+str(i)] ** 2
            for i in range(1, D):
                for j in range(len(df_space_vector)):
                    df_space_vector['Q*D'+str(i)][j] = df_space_vector['Q' +
                                                                    chr(178)][j] * df_space_vector['D'+str(i) + chr(178)][j]
            st.table(df_space_vector)
            for i in range(1, D):
                st.latex(
                    r'''Q \cdot D''' + str(i) + r''' = ''' +
                    str(round(df_space_vector['Q*D' + str(i)].sum(), 4)) + r''' '''
                )

            st.subheader("Perhitungan Cosine Similarity")
            st.latex(r'''\text{Cosine}\;\theta_{D_i}=\frac{\sum Q \cdot D_i}{\sqrt{\sum Q^2} \cdot \sqrt{\sum D_i^2}}''')
            df_cosine = pd.DataFrame(index=['Cosine'], columns=[
                'D'+str(i) for i in range(1, D)])
            for i in range(1, D):
                st.latex(
                    r'''Cosine\;\theta_{D''' + str(i) + r'''}=\frac{''' + str(round(df_space_vector['Q*D' + str(i)].sum(), 4)) + '''}{''' + str(sqrt_q) + ''' * ''' + str(sqrt_d[i-1]) + '''}= ''' + str(round(df_space_vector['Q*D' + str(i)].sum() / (sqrt_q * sqrt_d[i-1]), 4)) + r'''''')
                df_cosine['D'+str(i)] = df_space_vector['Q*D' +
                                                        str(i)].sum() / (sqrt_q * sqrt_d[i-1])
            st.table(df_cosine)

            # Menghitung dan menampilkan peringkat dokumen
            cosine_sim = {i: df_cosine['D'+str(i)].values[0] for i in range(1, D)}
            ranked_docs = sorted(cosine_sim, key=cosine_sim.get, reverse=True)

            st.subheader("Ranking Dokumen")
            df_ranking = pd.DataFrame({
                'Dokumen': ranked_docs,
                'Cosine Similarity': [cosine_sim[d] for d in ranked_docs]
            })
            st.table(df_ranking)