from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


class Months:
    MONTHS_WORDS1 = {"January": "01", "February": "02", "March": "03", "April": "04", "May": "05", "June": "06", "July": "07", "August": "08", "September": "09", "October": "10", "November": "11", "December": "12"}
    MONTHS_WORDS2 = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"}


class Translation:
    HEBREW_TO_ENGLISH = pipeline(task = "translation", model = "Helsinki-NLP/opus-mt-tc-big-he-en")
    RUSSIAN_TO_ENGLISH = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")