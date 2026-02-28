from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from detoxify import Detoxify



class Websites:
    WEBSITES_UKRAINE_RUSSIA = ["kpru", "ukpravda", "rt", "liganet"]
    WEBSITES_PALESTINE_ISRAEL = ["jpost", "alquds", "ynet", "ynet_global"]


class Months:
    MONTHS_WORDS1 = {"January": "01", "February": "02", "March": "03", "April": "04", "May": "05", "June": "06", "July": "07", "August": "08", "September": "09", "October": "10", "November": "11", "December": "12"}
    MONTHS_WORDS2 = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"}
    MONTHS_RUSSIAN = {"января": "01", "февраля": "02", "марта": "03", "апреля": "04","мая": "05", "июня": "06", "июля": "07", "августа": "08","сентября": "09", "октября": "10", "ноября": "11", "декабря": "12"}


class PretrainedModels:
    HEBREW_TO_ENGLISH = pipeline(task = "translation_he_to_en", model = "Helsinki-NLP/opus-mt-tc-big-he-en")
    RUSSIAN_TO_ENGLISH = pipeline(task = "translation_ru_to_en", model = "Helsinki-NLP/opus-mt-ru-en")

    SENTENCE_EMBEDDER = SentenceTransformer(model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2")
    FILTERS_UKRAINE_RUSSIA = ["Russia Ukraine conflict", "Ukraine Russia relations", "Eastern Europe crisis"]
    FILTERS_PALESTINE_ISRAEL = ["Israel Palestine conflict", "Gaza West Bank situation", "Middle East political tensions"]

    TOXICITY_DETECTOR = Detoxify(model_type = "original")

    EMOTION_CLASSIFIER = pipeline(task = "text-classification", model = "j-hartmann/emotion-english-distilroberta-base", return_all_scores = True)