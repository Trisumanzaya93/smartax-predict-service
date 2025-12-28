import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# =========================
# SETUP NLTK
# =========================
nltk.download("stopwords", quiet=True)

# =========================
# IMPORTANT WORDS (JANGAN DIHAPUS)
# =========================
IMPORTANT_WORDS = {
    "tidak",
    "belum",
    "tahu",
    "paham",
    "mengerti",
    "pembetulan",
    "pajak",
    "cara"
}

# =========================
# STOPWORDS (CUSTOM)
# =========================
stop_words = set(stopwords.words("indonesian"))
stop_words = stop_words - IMPORTANT_WORDS

# =========================
# STEMMER
# =========================
stemmer = StemmerFactory().create_stemmer()

# =========================
# NEGATION HANDLING
# =========================
def handle_negation(tokens):
    """
    Gabungkan negasi:
    tidak tahu -> tidak_tahu
    belum paham -> belum_paham
    """
    result = []
    i = 0
    while i < len(tokens):
        if tokens[i] in {"tidak", "belum"} and i + 1 < len(tokens):
            result.append(f"{tokens[i]} {tokens[i + 1]}")
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result

# =========================
# MAIN PREPROCESS FUNCTION
# =========================
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # 1. lowercase
    text = text.lower()

    # 2. remove angka & simbol
    text = re.sub(r"[^a-z\s]", " ", text)

    # 3. normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 4. tokenizing
    tokens = text.split()

    # 5. negation handling (SEBELUM stopword)
    tokens = handle_negation(tokens)

    # 6. stopword removal
    tokens = [t for t in tokens if t not in stop_words]

    # 7. stemming (skip token dengan underscore)
    tokens = [
        stemmer.stem(t) if "_" not in t and "perpajakan" not in t else t
        for t in tokens
    ]

    return " ".join(tokens)

