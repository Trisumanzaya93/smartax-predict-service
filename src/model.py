import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from preprocess import preprocess_text

# ======================
# 1. LOAD DATA
# ======================
df = pd.read_csv("data/Daftar_Masalah_WP.csv")

df["clean_text"] = df["Permasalahan Wajib Pajak"].apply(preprocess_text)

# ======================
# 2. TF-IDF
# ======================
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.85,
    sublinear_tf=True
)

X = vectorizer.fit_transform(df["clean_text"])

# ======================
# 3. KMEANS
# ======================
N_CLUSTERS = 6

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(X)

# ======================
# 4. 🔥 SIMPAN MODEL (INI YANG KURANG)
# ======================
joblib.dump(vectorizer, "model/tfidf.pkl")
joblib.dump(kmeans, "model/kmeans.pkl")

print("✅ MODEL DISIMPAN")

# ======================
# 5. (OPTIONAL) DEBUG CLUSTER
# ======================
for i in range(N_CLUSTERS):
    print(f"\n=== CLUSTER {i} ===")
    samples = df[df["cluster"] == i]["Permasalahan Wajib Pajak"]
    for s in samples:
        print("-", s)
