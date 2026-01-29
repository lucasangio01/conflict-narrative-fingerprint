import re
import pandas as pd



def split_sentences(text):
    # Split on ., !, ? followed by whitespace
    # Keeps punctuation at the end of sentence
    return re.split(r'(?<=[.!?])\s+', text.strip())


def fix_spaced_date(text):
    if pd.isna(text):
        return text
    text = str(text)
    text = re.sub(r'(?<!\w)(?:[A-Z]\s){2,}[A-Z]', lambda m: m.group(0).replace(" ", ""), text)
    text = re.sub(r'(?<!\d)(?:\d\s){1,}\d', lambda m: m.group(0).replace(" ", ""), text)
    text = re.sub(r'\s*:\s*', ":", text)
    text = re.sub(r'\s+,', ",", text)
    text = re.sub(r'\s{2,}', " ", text).strip()
    return text
