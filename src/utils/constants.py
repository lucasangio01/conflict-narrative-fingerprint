from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from detoxify import Detoxify
from fastcoref import LingMessCoref, FCoref


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

    COREF_LINGMESS = LingMessCoref(device)
    COREF_F = FCoref(device)


class Axis:
    AXIS_SEEDS = {"competence": {"high": ["powerful", "strategic", "efficient", "capable", "advanced", "strong", "organized"], "low": ["weak", "failing", "disorganized", "incompetent", "chaotic", "vulnerable", "ineffective"]}, "morality": {"high": ["righteous", "just", "innocent", "heroic", "moral", "civilized", "peaceful"], "low": ["cruel", "evil", "terrorist", "murderous", "corrupt", "barbaric", "aggressive"]}}


class Verbs:
    VIOLENT_VERBS = ["attack", "bomb", "kill", "destroy", "invade", "strike", "shell", "target", "raid", "fire"]
    MODAL_VERBS = {"may", "might", "could", "would", "should", "can"}
    HEDGES = {"seem", "appear", "likely", "possible", "allege", "probable", "suggest", "warn", "claim", "reportedly"}


class NamesDicts:
    SYNONYM_MAP = {
    # --- USA / WEST ---
    "the united states": "usa", "u.s.": "usa", "us": "usa", "u.s.u.": "usa",
    "washington": "usa", "america": "usa", "state department": "usa",
    "the white house": "usa", "brussels": "eu", "european union": "eu",

    # --- UKRAINE / RUSSIA ---
    "kyiv": "kiev", "vladimir zelensky": "zelensky", "volodymyr zelensky": "zelensky",
    "andriy yermak": "yermak", "andriy yermak's": "yermak",
    "the office of the president": "ukraine", "office of the president": "ukraine",
    "zaporizhzhia": "zaporizhzhia", "crimea": "crimea", "kharkiv": "kharkiv", "donetsk": "donetsk",
    "vladimir putin": "putin", "russian federation": "russia", "the kremlin": "kremlin", "ussr": "russia",
    "warsaw": "poland", "berlin": "germany", "paris": "france",

    # --- ISRAEL / PALESTINE ---
    "the state of israel": "israel", "jerusalem": "israel", "the knesset": "israel",
    "benjamin netanyahu": "netanyahu", "bibi": "netanyahu",
    "israeli defense forces": "idf", "iaf": "idf", "israeli army": "idf", "occupation forces": "idf",
    "tel aviv": "israel", "judea and samaria": "israel", "samaria": "israel", "the zionist entity": "israel",
    "the gaza strip": "gaza", "gaza strip": "gaza", "the strip": "gaza",
    "mahmoud abbas": "abbas", "abu mazen": "abbas", # PA Leader
    "mansour abbas": "mansour abbas", # Israeli-Arab Politician (Ra'am)
    "palestinian authority": "pa", "hamas movement": "hamas", "yahya sinwar": "sinwar",
    "ismail haniyeh": "haniyeh", "the resistance": "militants", "fighters": "militants",
    "the west bank": "palestine", "west bank": "palestine", "the state of palestine": "palestine",
    "east jerusalem": "palestine", "the palestinian authority": "pa", "authority": "pa",
    "oslo": "pa", "unrwa": "un", "court": "un", "türkiye": "turkey",

    # --- IRAN AXIS ---
    "the islamic republic": "iran", "tehran": "iran", "irgc": "iran", "khamenei": "iran",}

    RU_UK_BASE = {
        "russia": "RU_POLITICAL", "putin": "RU_POLITICAL", "moscow": "RU_POLITICAL", "kremlin": "RU_POLITICAL",
        "ukraine": "UKR_POLITICAL", "zelensky": "UKR_POLITICAL", "kiev": "UKR_POLITICAL", "yermak": "UKR_POLITICAL",
        "zaporizhzhia": "UKR_POLITICAL", "crimea": "UKR_POLITICAL", "kharkiv": "UKR_POLITICAL", "donetsk": "UKR_POLITICAL",
        "usa": "WEST_ACTORS", "nato": "WEST_ACTORS", "eu": "WEST_ACTORS", "biden": "WEST_ACTORS", "trump": "WEST_ACTORS",
        "poland": "WEST_ACTORS", "germany": "WEST_ACTORS", "france": "WEST_ACTORS",
        "army": "RU_MILITARY", "wagner": "RU_MILITARY", "afu": "UKR_MILITARY", "azov": "UKR_MILITARY",
        "china": "INTL_ACTORS", "un": "INTL_ACTORS", "iaea": "INTL_ACTORS", "turkey": "INTL_ACTORS",
        "civilians": "CIVILIANS"
    }

    IZ_PA_BASE = {
        # Israel
        "netanyahu": "ISR_POLITICAL", "israel": "ISR_POLITICAL", "knesset": "ISR_POLITICAL",
        "mansour abbas": "ISR_POLITICAL", "gallant": "ISR_POLITICAL", "gantz": "ISR_POLITICAL",
        "idf": "ISR_MILITARY", "mossad": "ISR_MILITARY", "shin bet": "ISR_MILITARY",

        # Palestine
        "abbas": "PAL_POLITICAL", "pa": "PAL_POLITICAL", "palestine": "PAL_POLITICAL",
        "hamas": "PAL_ORG", "haniyeh": "PAL_ORG", "sinwar": "PAL_ORG", "pij": "PAL_ORG",
        "militants": "PAL_RESISTANCE", "qassam": "PAL_RESISTANCE", "gaza": "PAL_RESISTANCE",

        # International
        "usa": "INTL_ACTORS", "un": "INTL_ACTORS", "biden": "INTL_ACTORS", "trump": "INTL_ACTORS",
        "iran": "INTL_ACTORS", "lebanon": "INTL_ACTORS", "turkey": "INTL_ACTORS", "syria": "INTL_ACTORS",
        "saudi arabia": "INTL_ACTORS", "qatar": "INTL_ACTORS", "egypt": "INTL_ACTORS", "hezbollah": "INTL_ACTORS",

        # Civil / Other
        "civilians": "CIVILIANS", "hostages": "CIVILIANS", "settlers": "SETTLERS"
    }
