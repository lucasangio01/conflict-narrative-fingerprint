from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


class Months:
    MONTHS_WORDS1 = {"January": "01", "February": "02", "March": "03", "April": "04", "May": "05", "June": "06", "July": "07", "August": "08", "September": "09", "October": "10", "November": "11", "December": "12"}
    MONTHS_WORDS2 = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"}
    MONTHS_RUSSIAN = {"января": "01", "февраля": "02", "марта": "03", "апреля": "04","мая": "05", "июня": "06", "июля": "07", "августа": "08","сентября": "09", "октября": "10", "ноября": "11", "декабря": "12"}


class Translation:
    HEBREW_TO_ENGLISH = pipeline(task = "translation_he_to_en", model = "Helsinki-NLP/opus-mt-tc-big-he-en")
    RUSSIAN_TO_ENGLISH = pipeline(task = "translation_ru_to_en", model = "Helsinki-NLP/opus-mt-ru-en")