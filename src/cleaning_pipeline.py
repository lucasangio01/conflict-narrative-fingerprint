import pandas as pd
from utils.constants import Months
import logging



logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)


class CleaningPipeline:
    def __init__(self, website):
        self.website = website
        if self.website in ["jpost", "ynet", "ynet_global"]:
            folder1 = "israel"
        elif self.website == "alquds":
            folder1 = "palestine"
        elif self.website in ["kpru", "rt"]:
            folder1 = "russia"
        elif self.website in ["liganet", "ukpravda"]:
            folder1 = "ukraine"

        self.csv_path_original = f"../data/{folder1}/original/{self.website}.csv"
        self.csv_path_cleaned_date = f"../data/{folder1}/cleaned_date/{self.website}.csv"
    

    def remove_spaces_russia_kp(self):
        if self.website != "kpru":
            return
        else:
            logging.info("Removing blank spaces from KP (Russia)...")
            df = pd.read_csv(self.csv_path_original)
            df["title"] = df["title"].str.replace(r"(?<=\w)\s(?=\w)", "", regex = True).str.replace(r"\s+([,.:;!?])", r"\1", regex = True).str.replace(r"([,.:;!?])(?=\w)", r"\1 ", regex = True).str.replace(r"\s{2,}", " ", regex = True).str.strip()
            df["text"] = df["text"].str.replace(r"(?<=\w)\s(?=\w)", "", regex = True).str.replace(r"\s+([,.:;!?])", r"\1", regex = True).str.replace(r"([,.:;!?])(?=\w)", r"\1 ", regex = True).str.replace(r"\s{2,}", " ", regex = True).str.strip()
            df = df.drop(columns=["Unnamed: 0"])
            df.to_csv(self.csv_path_cleaned_date)
            logging.info("Blank spaces removed from KP (Russia)!")


    def clean_date(self):
        logging.info(f"Cleaning dates for {self.website}...")   
        self.df_original = pd.read_csv(self.csv_path_original) 
        pattern = r"\b(" + "|".join(Months.MONTHS_WORDS1.keys()) + r")\b"
        self.df_original["date"] = self.df_original["date"].str.replace(pattern, lambda m: Months.MONTHS_WORDS1[m.group(0)], regex=True).str.split(",", n = 1).str[0].str.replace("['", "").str.strip().str.replace(" ", "-")
        self.df_original = self.df_original.drop(columns=["Unnamed: 0"])
        self.df_cleaned_date = self.df_original
        self.df_cleaned_date.to_csv(self.csv_path_cleaned_date)
        logging.info(f"Dates cleaned for {self.website}!")
    


if __name__ == "__main__":
    cleaning_pipeline = CleaningPipeline(website = "kpru")
    # cleaning_pipeline.clean_date()
    cleaning_pipeline.remove_spaces_russia_kp()
    
