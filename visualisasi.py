import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# load dataset
df = pd.read_csv("OnlineRetail.csv", encoding='ISO-8859-1')

st.title("Market Basket Analysis")

# Fungsi untuk membaca item dari file teks
def load_items_from_file(file_path):
    with open(file_path, 'r') as file:
        items = file.read()
        items_list = eval(items)  # Evaluasi string menjadi list
    return items_list

# Fungsi Streamlit untuk menampilkan selectbox
def user_input_item():
    # Memuat item dari file teks
    items = load_items_from_file('unique_items.txt')
    item = st.selectbox("Item", items)
    # st.write(f"Selected item: {item}")
    return item

item = user_input_item()

# Set transactions
gp_invoiceno = df.groupby('InvoiceNo')
transactions = []
for name,group in gp_invoiceno:
    transactions.append(list(group['Description'].map(str)))

# Membuat DataFrame dengan one-hot encoding
te = TransactionEncoder()  # Membuat instance TransactionEncoder
te_ary = te.fit(transactions).transform(transactions)  # Melakukan fitting dan transformasi pada transaksi
df = pd.DataFrame(te_ary, columns=te.columns_)  # Mengonversi hasil one-hot encoding menjadi DataFrame

frequent_itemsets = fpgrowth(df, min_support=0.01, use_colnames=True)

# Menghasilkan aturan asosiasi
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
rules.sort_values('confidence', ascending=False, inplace=True)

def parse_list(x):
    x = list(x)
    if len(x) == 1:
        return x[0]
    elif len(x) > 1:
        return ", ".join(x)

def return_item_df(item_antecedents):
    data = rules[["antecedents", "consequents"]].copy()

    data["antecedents"] = data["antecedents"].apply(parse_list)
    data["consequents"] = data["consequents"].apply(parse_list)

    filtered_data = data.loc[data["antecedents"] == item_antecedents]
    print(filtered_data)  # Periksa hasil filter

    # Cek apakah ada baris yang memenuhi kondisi pemfilteran
    if not filtered_data.empty:
        # Jika ada, kembalikan nilai dari baris pertama dalam bentuk list
        return list(filtered_data.iloc[0,:])
    else:
        # Jika tidak ada, berikan pesan bahwa tidak ada item yang cocok
        return f"Tidak ada rekomendasi untuk item '{item_antecedents}'"
        


st.markdown("Hasil Rekomendasi : ")
st.success(f"Jika Konsumen Membeli **{item}**, maka Membeli **{return_item_df(item)[1]}** Secara Bersamaan")

