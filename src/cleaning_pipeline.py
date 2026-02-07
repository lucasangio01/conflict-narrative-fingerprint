import pandas as pd
import re
import torch
from transformers import pipeline
import logging
from src.utils.constants import Months


logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)


class Translation:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        if self.device == 0:
            logging.info("Using GPU for translation")
        else:
            logging.info("GPU not found, using CPU")
        self.models = {"hebrew": pipeline("translation", model = "Helsinki-NLP/opus-mt-tc-big-he-en", device = self.device), "russian": pipeline("translation", model = "Helsinki-NLP/opus-mt-ru-en", device = self.device)}


    def batch_translate(self, texts, language, batch_size = 32):
        language = language.lower()
        if language == "english":
            return texts

        translator = self.models.get(language)
        if translator is None:
            raise ValueError(f"No translator available for language: {language}")

        translated_texts = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            translated_batch = translator(batch, max_length=512, truncation=True)
            translated_texts.extend(t["translation_text"] for t in translated_batch)
        return translated_texts



class CleaningPipeline:
    def __init__(self, website):
        self.website = website
        if website in ["jpost", "ynet", "ynet_global"]:
            self.original_language = "hebrew" if website == "ynet" else "english"
        elif website in ["kpru", "rt"]:
            self.original_language = "russian"
        else:
            self.original_language = "english"

        self.csv_path_original = f"{website}_original.csv"
        self.csv_path_chunked = f"{website}_chunked.csv"


    def clean_date(self):
        self.df = pd.read_csv(self.csv_path_original)
        pattern_en = r"\b(" + "|".join(Months.MONTHS_WORDS1.keys()) + r")\b"
        pattern_ru = r"\b(" + "|".join(Months.MONTHS_RUSSIAN.keys()) + r")\b"

        self.df["date"] = (
            self.df["date"]
            .str.replace(pattern_ru, lambda m: Months.MONTHS_RUSSIAN[m.group(0)], regex=True)
            .str.replace(pattern_en, lambda m: Months.MONTHS_WORDS1[m.group(0)], regex=True)
            .str.replace(" ", "-", regex=False))

        self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce").dt.strftime("%d-%m-%Y")
        self.df = self.df.drop(columns=["Unnamed: 0"], errors="ignore")



class Pipeline:
    def __init__(self, df, language, max_chars=800):
        self.df = df
        self.language = language
        self.MAX_CHARS = max_chars
        self.translator = Translation()


    @staticmethod
    def split_sentences(text):
        return re.split(r'(?<=[.!?])\s+', str(text).strip())


    def translate_titles(self):
        if "title" in self.df.columns:
            self.df["title"] = self.translator.batch_translate(self.df["title"].tolist(), self.language)


    def chunk_and_translate_text(self, save_path):
        rows = []

        for _, row in self.df.iterrows():
            title, date, text = row["title"], row["date"], row["text"]
            sentences = self.split_sentences(text)

            current = ""
            for sent in sentences:
                if len(current) + len(sent) > self.MAX_CHARS:
                    translated = self.translator.batch_translate([current], self.language)[0]
                    rows.append({"title": title, "date": date, "text": translated})
                    current = sent
                else:
                    current += " " + sent

            if current.strip():
                translated = self.translator.batch_translate([current], self.language)[0]
                rows.append({"title": title, "date": date, "text": translated})

        pd.DataFrame(rows).to_csv(save_path, index=False)
        logging.info(f"Chunked & translated CSV saved to {save_path}")



if __name__ == "__main__":
    website = "rt"
    cleaner = CleaningPipeline(website)
    cleaner.clean_date()
    pipe = Pipeline(df = cleaner.df, language = cleaner.original_language)
    pipe.translate_titles()
    pipe.chunk_and_translate_text(cleaner.csv_path_chunked)
    