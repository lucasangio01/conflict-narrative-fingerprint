import pandas as pd
import re
import logging
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.utils.constants import Months, PretrainedModels

logging.basicConfig(level=logging.INFO)


class Translation:
    def __init__(self, language):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.tokenizers = {}
        self.language = language.lower()

        model_paths = PretrainedModels.TRANSLATION_MODEL_PATHS

        if self.language in model_paths:
            path = model_paths[self.language]
            logging.info(f"Loading {self.language} model: {path}")

            self.tokenizers[self.language] = AutoTokenizer.from_pretrained(path)
            self.models[self.language] = AutoModelForSeq2SeqLM.from_pretrained(path).to(self.device)


    def batch_translate(self, texts, batch_size=16):
        if self.language == "english" or not texts:
            return texts

        model = self.models.get(self.language)
        tokenizer = self.tokenizers.get(self.language)

        if model is None or tokenizer is None:
            raise ValueError(f"No model/tokenizer loaded for: {self.language}")

        translated_texts = []

        for i in range(0, len(texts), batch_size):
            batch = [str(t) for t in texts[i:i + batch_size] if str(t).strip() != ""]
            if not batch: continue

            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)

            with torch.no_grad():
                translated_tokens = model.generate(**inputs)

            decoded = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            translated_texts.extend(decoded)

        return translated_texts


class CleaningPipeline:
    def __init__(self, website):
        self.website = website
        if self.website in ["jpost", "ynet", "ynet_global"]:
            self.original_language = "hebrew" if self.website == "ynet" else "english"
        elif self.website in ["kpru", "rt"]:
            self.original_language = "russian"
        else:
            self.original_language = "english"
        self.csv_path_original = f"1_{self.website}_original.csv"

    def clean_date(self):
            self.df = pd.read_csv(self.csv_path_original)

            if self.website == "jpost":
                self.df["date"] = self.df["date"].astype(str).str.replace(r'(?<=\w)\s(?=\w)', '', regex=True)

            pattern_en = r"\b(" + "|".join(Months.MONTHS_WORDS1.keys()) + r")\b"
            pattern_ru = r"\b(" + "|".join(Months.MONTHS_RUSSIAN.keys()) + r")\b"
            pattern_days = r"\b(" + "|".join(Months.DAYS_TO_REPLACE) + r")\b"
            pattern_abbr = r"\b(" + "|".join(Months.MONTHS_WORDS_ABBR.keys()) + r")\b"

            self.df["date"] = (self.df["date"].astype(str)
                .str.replace(pattern_days, "", regex=True, flags=re.IGNORECASE)
                .str.replace(pattern_en, lambda m: Months.MONTHS_WORDS1[m.group(0).title()], regex=True, flags=re.IGNORECASE)
                .str.replace(pattern_ru, lambda m: Months.MONTHS_RUSSIAN[m.group(0).lower()], regex=True, flags=re.IGNORECASE)
                .str.replace(pattern_abbr, lambda m: Months.MONTHS_WORDS_ABBR[m.group(0).title()], regex=True, flags=re.IGNORECASE))

            self.df["date"] = (self.df["date"]
                .str.replace(r"[,/]", " ", regex=True)
                .str.replace(r"\s+\d{1,2}\s*:\s*\d{2}.*$", "", regex=True)
                .str.replace(r"\s+", "-", regex=True)
                .str.strip("-"))

            self.df["date"] = self.df["date"].apply(lambda x: x + "-2026" if isinstance(x, str) and x.count("-") == 1 else x)
            self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce", dayfirst=True).dt.strftime("%d-%m-%Y")

            return self.df


class TranslationPipeline:
    def __init__(self, df, language):
        self.df = df
        self.language = language
        self.translator = Translation(language)


    @staticmethod
    def split_sentences(text):
        return re.split(r'(?<=[.!?])\s+', str(text).strip())


    def translate_full_text(self):
        self.df = self.df.dropna(subset=['text']).copy()
        if 'title' in self.df.columns:
            print(f"🏷️ Translating {len(self.df)} titles...")
            titles = self.df['title'].fillna("").astype(str).tolist()
            self.df["title_en"] = self.translator.batch_translate(titles)
        translated_articles = []
        texts = self.df['text'].dropna().astype(str).tolist()

        print(f"🌍 Translating {len(texts)} articles via Manual Seq2Seq...")
        for article in tqdm(texts):
            sentences = self.split_sentences(article)
            translated_sents = self.translator.batch_translate(sentences)
            translated_articles.append(" ".join(translated_sents))

        self.df["text_en"] = translated_articles
        return self.df


def main(website="alquds"):
    cleaner = CleaningPipeline(website)
    df_cleaned = cleaner.clean_date()

    pipe = TranslationPipeline(df=df_cleaned, language=cleaner.original_language)
    df_translated = pipe.translate_full_text()

    df_translated.to_csv(f"2_{website}_english.csv", index=False)
    print("✅ Success! Translation and cleaning complete.")


if __name__ == "__main__":
    main()
