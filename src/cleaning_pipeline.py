import pandas as pd
from utils.constants import Months, Translation
from utils.other import split_sentences, fix_spaced_date
import logging
import re


logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)


class CleaningPipeline:
    def __init__(self, website):
        self.website = website
        self.MAX_CHUNK_CHARACTERS = 800
        self.chunked_data = []

        if self.website in ["jpost", "ynet", "ynet_global"]:
            self.folder1 = "israel"
            if self.website == "ynet":
                self.original_language = "hebrew"
            else:
                self.original_language = "english"
        elif self.website == "alquds":
            self.folder1 = "palestine"
            self.original_language = "english"
        elif self.website in ["kpru", "rt"]:
            self.folder1 = "russia"
            self.original_language = "russian"
        elif self.website in ["liganet", "ukpravda"]:
            self.folder1 = "ukraine"
            self.original_language = "english"

        self.csv_path_original = f"../data/{self.folder1}/1_original/{self.website}.csv"
        self.csv_path_cleaned_date = f"../data/{self.folder1}/2_cleaned_date/{self.website}.csv"
        self.csv_path_title_translated = f"../data/{self.folder1}/3_title_translated/{self.website}.csv"
        self.csv_path_chunked_text = f"../data/{self.folder1}/4_chunked_text/{self.website}.csv"
    

    def clean_date(self):
        logging.info(f"Cleaning dates for {self.website}...")   

        self.df_original = pd.read_csv(self.csv_path_original) 
        self.df_original["date"] = self.df_original["date"].apply(fix_spaced_date)
        date_pattern1 = r"\b(" + "|".join(Months.MONTHS_WORDS1.keys()) + r")\b"
        date_pattern2 = r"\b(" + "|".join(Months.MONTHS_RUSSIAN.keys()) + r")\b"
        
        if self.website != "jpost":
            self.df_original["date"] = self.df_original["date"].str.replace(date_pattern2, lambda m: Months.MONTHS_RUSSIAN[m.group(0)], regex=True).str.split(",", n = 1).str[0].str.replace("['", "").str.strip().str.replace(" ", "-")       
            self.df_original["date"] = self.df_original["date"].str.replace(date_pattern1, lambda m: Months.MONTHS_WORDS1[m.group(0)], regex=True).str.split(",", n = 1).str[0].str.replace("['", "").str.strip().str.replace(" ", "-")
        
        self.df_original["date"] = pd.to_datetime(self.df_original["date"], errors = "coerce").dt.strftime("%d-%m-%Y")
        if "Unnamed: 0" in self.df_original.columns:
            self.df_original = self.df_original.drop(columns = ["Unnamed: 0"])
        self.df_cleaned_date = self.df_original

        logging.info(f"Dates cleaned for {self.website}!")


    def remove_spaces(self):
        if self.website not in ["kpru", "jpost"]:
            return
        else:
            logging.info("Removing blank spaces...")

            self.df_cleaned_date["date"] = self.df_cleaned_date["date"].str.replace(r"(?<=\w)\s(?=\w)", "", regex = True).str.replace(r"\s+([,.:;!?])", r"\1", regex = True).str.replace(r"([,.:;!?])(?=\w)", r"\1 ", regex = True).str.replace(r"\s{2,}", " ", regex = True).str.strip()
            if self.website == "kpru":
                self.df_cleaned_date["title"] = self.df_cleaned_date["title"].str.replace(r"(?<=\w)\s(?=\w)", "", regex = True).str.replace(r"\s+([,.:;!?])", r"\1", regex = True).str.replace(r"([,.:;!?])(?=\w)", r"\1 ", regex = True).str.replace(r"\s{2,}", " ", regex = True).str.strip()
                self.df_cleaned_date["text"] = self.df_cleaned_date["text"].str.replace(r"(?<=\w)\s(?=\w)", "", regex = True).str.replace(r"\s+([,.:;!?])", r"\1", regex = True).str.replace(r"([,.:;!?])(?=\w)", r"\1 ", regex = True).str.replace(r"\s{2,}", " ", regex = True).str.strip()
            if "Unnamed: 0" in self.df_cleaned_date.columns:
                self.df_cleaned_date = self.df_cleaned_date.drop(columns = ["Unnamed: 0"])

            logging.info("Blank spaces removed!")
    

    def title_translation(self):
        logging.info(f"Translating {self.website} titles into English...")

        self.df_title_translated = self.df_cleaned_date
        translations = [Translation.HEBREW_TO_ENGLISH(title)[0]["translation_text"] if self.original_language == "hebrew" else Translation.RUSSIAN_TO_ENGLISH(title)[0]["translation_text"] if self.original_language == "russian" else title for title in self.df_title_translated['title']]
        self.df_title_translated["title"] = translations
        if "Unnamed: 0" in self.df_title_translated.columns:
            self.df_title_translated = self.df_title_translated.drop(columns = ["Unnamed: 0"])

        logging.info(f"{self.website} titles translated!")


    def chunk_text(self):
        logging.info(f"Chunking text for {self.website}...")

        df = self.df_title_translated
        self.chunked_data = []

        for _, row in df.iterrows():
            title = row["title"]
            date = row["date"]
            text = str(row["text"])

            sentences = split_sentences(text)
            current_chunk = ""

            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 > self.MAX_CHUNK_CHARACTERS:
                    if current_chunk.strip():
                        self.chunked_data.append({"title": title, "date": date, "text": current_chunk.strip()})
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence

            if current_chunk.strip():
                self.chunked_data.append({"title": title, "date": date, "text": current_chunk.strip()})
        df_chunked = pd.DataFrame(self.chunked_data)
        df_chunked.to_csv(self.csv_path_chunked_text, index=False)
        logging.info(f"{self.website} text split into chunks!")



if __name__ == "__main__":
    cleaning_pipeline = CleaningPipeline(website = "liganet")
    cleaning_pipeline.clean_date()
    cleaning_pipeline.remove_spaces()
    cleaning_pipeline.title_translation()
    cleaning_pipeline.chunk_text()
    
