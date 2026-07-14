import functools


class Websites:
    WEBSITES_UKRAINE_RUSSIA = ["kpru", "ukpravda", "rt", "liganet"]
    WEBSITES_PALESTINE_ISRAEL = ["jpost", "alquds", "ynet", "ynet_global"]

    DISPLAY_NAMES = {
        "kpru":        "Komsomolskaya Pravda",
        "ukpravda":    "Ukrainska Pravda",
        "rt":          "Russia Today",
        "liganet":     "Liga.net",
        "alquds":      "Al-Quds",
        "jpost":       "Jerusalem Post",
        "ynet":        "Ynet",
        "ynet_global": "Ynet Global",
    }


class Months:
    MONTHS_WORDS1 = {"January": "01", "February": "02", "March": "03", "April": "04", "May": "05", "June": "06", "July": "07", "August": "08", "September": "09", "October": "10", "November": "11", "December": "12"}
    MONTHS_WORDS_ABBR = {"Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06", "Jul": "07", "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"}
    MONTHS_RUSSIAN = {"января": "01", "февраля": "02", "марта": "03", "апреля": "04","мая": "05", "июня": "06", "июля": "07", "августа": "08","сентября": "09", "октября": "10", "ноября": "11", "декабря": "12"}
    DAYS_TO_REPLACE = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


class PretrainedModels:
    """
    Heavy models -- and the packages that define them -- are loaded lazily
    (on first call, then cached) rather than at module import time. Importing
    this module just for entity dicts / word lists (which every analysis
    script does) must not eat the multi-second import cost of transformers/
    sentence_transformers/detoxify/fastcoref, let alone force-load a model.
    """
    FILTERS_UKRAINE_RUSSIA = ["Russia Ukraine conflict", "Ukraine Russia relations", "Eastern Europe crisis"]
    FILTERS_PALESTINE_ISRAEL = ["Israel Palestine conflict", "Gaza West Bank situation", "Middle East political tensions"]

    # Model identifiers reused across scripts -- kept as plain strings (not
    # lazy-loaded) since referencing a name has no import/loading cost.
    SPACY_MODEL_LG = "en_core_web_lg"   # ships a parser; used wherever sentence segmentation matters
    SPACY_MODEL_SM = "en_core_web_sm"   # smaller/faster base for the coref pipeline (parser excluded there)
    SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    SENTENCE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    TRANSLATION_MODEL_PATHS = {"hebrew": "Helsinki-NLP/opus-mt-tc-big-he-en", "russian": "Helsinki-NLP/opus-mt-ru-en"}

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def hebrew_to_english():
        from transformers import pipeline
        return pipeline(task="translation_he_to_en", model=PretrainedModels.TRANSLATION_MODEL_PATHS["hebrew"])

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def russian_to_english():
        from transformers import pipeline
        return pipeline(task="translation_ru_to_en", model=PretrainedModels.TRANSLATION_MODEL_PATHS["russian"])

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def sentence_embedder():
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name_or_path=PretrainedModels.SENTENCE_EMBEDDING_MODEL)

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def toxicity_detector():
        from detoxify import Detoxify
        return Detoxify(model_type="original")

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def emotion_classifier():
        from transformers import pipeline
        return pipeline(task="text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def coref_lingmess():
        import torch
        from fastcoref import LingMessCoref
        return LingMessCoref(device="cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def coref_fcoref():
        import torch
        from fastcoref import FCoref
        return FCoref(device="cuda" if torch.cuda.is_available() else "cpu")


class Axis:
    AXIS_SEEDS = {
        "competence": {"high": ["powerful", "strategic", "efficient", "capable", "advanced", "strong", "organized"], "low": ["weak", "failing", "disorganized", "incompetent", "chaotic", "vulnerable", "ineffective"]},
        "morality": {"high": ["righteous", "just", "innocent", "heroic", "moral", "civilized", "peaceful"], "low": ["cruel", "evil", "terrorist", "murderous", "corrupt", "barbaric", "aggressive"]},
    }


class Verbs:
    VIOLENT_VERBS = {
        "attack", "bomb", "kill", "destroy", "invade", "strike", "shell", "target", "raid", "fire",
        "shoot", "assassinate", "massacre", "wound", "besiege", "siege", "execute", "detain", "arrest", "expel",
    }
    MODAL_VERBS = {"may", "might", "could", "would", "should", "can"}
    HEDGES = {"seem", "appear", "likely", "possible", "allege", "probable", "suggest", "warn", "claim", "reportedly", "allegedly", "apparently"}
    NEGATION_TOKENS = {"not", "never", "no", "neither", "nor", "deny", "refuse", "reject", "false", "untrue"}


class Lexicons:
    DEHUMAN_LEXICON = {
        "animal":   ["rat", "vermin", "beast", "animal", "hyena", "wolf", "parasite", "insect", "viper", "pig", "monkey"],
        "disease":  ["cancer", "plague", "virus", "infection", "tumor", "disease", "toxic", "poison", "pestilence"],
        "subhuman": ["barbaric", "monster", "scum", "demon", "savage", "primitive", "beastly", "uncivilized", "evil"],
    }
    DEHUMAN_WORDS = set(word for words in DEHUMAN_LEXICON.values() for word in words)
    DEHUMAN_CATEGORY = {word: cat for cat, words in DEHUMAN_LEXICON.items() for word in words}


class NamesDicts:
    # Canonical surface-form -> canonical key. Safe to be a superset across scripts:
    # a mapped key that isn't present in a given script's active theater dict below
    # simply never matches (each script gates on `if clean in active_entities`).
    SYNONYM_MAP = {
        # --- USA / WEST ---
        "the united states": "usa", "u.s.": "usa", "united states": "usa",
        "washington": "usa", "america": "usa", "state department": "usa",
        "the white house": "usa", "brussels": "eu", "european union": "eu",

        # --- UKRAINE / RUSSIA ---
        "kyiv": "kiev",
        "vladimir zelensky": "zelensky", "volodymyr zelensky": "zelensky",
        "andriy yermak": "yermak", "andriy yermak's": "yermak",
        "the office of the president": "ukraine", "office of the president": "ukraine",
        "vladimir putin": "putin",
        "russian federation": "russia", "the kremlin": "kremlin", "ussr": "russia",
        "warsaw": "poland", "berlin": "germany", "paris": "france",

        # --- ISRAEL / PALESTINE ---
        "the state of israel": "israel", "the knesset": "israel",
        "tel aviv": "israel", "judea and samaria": "israel",
        "the zionist entity": "israel",
        "benjamin netanyahu": "netanyahu", "bibi": "netanyahu",
        "israeli defense forces": "idf", "iaf": "idf",
        "israeli army": "idf", "occupation forces": "idf",
        "the gaza strip": "gaza", "gaza strip": "gaza", "the strip": "gaza",
        "mahmoud abbas": "abbas", "abu mazen": "abbas",
        "mansour abbas": "mansour_abbas",       # Israeli-Arab politician -- kept distinct from Mahmoud Abbas
        "palestinian authority": "pa", "the palestinian authority": "pa",
        "hamas movement": "hamas",
        "yahya sinwar": "sinwar", "ismail haniyeh": "haniyeh",
        "al-qassam brigades": "qassam", "izz ad-din al-qassam": "qassam",
        "the resistance": "militants", "armed groups": "militants", "fighters": "militants",
        "the west bank": "palestine", "west bank": "palestine",
        "the state of palestine": "palestine", "east jerusalem": "palestine",
        "oslo accords": "pa",
        "unrwa": "un", "united nations relief": "un",
        "international court of justice": "icj", "international court": "icj",
        "türkiye": "turkey",

        # --- IRAN AXIS ---
        "the islamic republic": "iran", "tehran": "iran",
        "irgc": "iran", "khamenei": "iran",
    }

    # Named units/individuals/organisations only -- no generic common nouns
    # ("army", "forces", "troops", "government"), which fire on any mention
    # of the concept rather than the specific actor.
    RU_UK_BASE = {
        "russia": "RU_POLITICAL", "putin": "RU_POLITICAL", "moscow": "RU_POLITICAL", "kremlin": "RU_POLITICAL", "lavrov": "RU_POLITICAL",
        "ukraine": "UKR_POLITICAL", "zelensky": "UKR_POLITICAL", "kiev": "UKR_POLITICAL", "yermak": "UKR_POLITICAL",
        "zaporizhzhia": "UKR_POLITICAL", "crimea": "UKR_POLITICAL", "kharkiv": "UKR_POLITICAL", "donetsk": "UKR_POLITICAL",
        "usa": "WEST_ACTORS", "nato": "WEST_ACTORS", "eu": "WEST_ACTORS", "biden": "WEST_ACTORS", "trump": "WEST_ACTORS",
        "poland": "WEST_ACTORS", "germany": "WEST_ACTORS", "france": "WEST_ACTORS",
        "wagner": "RU_MILITARY",
        "afu": "UKR_MILITARY", "azov": "UKR_MILITARY", "syrskyi": "UKR_MILITARY",
        "china": "INTL_ACTORS", "un": "INTL_ACTORS", "iaea": "INTL_ACTORS", "turkey": "INTL_ACTORS",
        "civilians": "CIVILIANS", "refugees": "CIVILIANS",
    }

    IZ_PA_BASE = {
        "netanyahu": "ISR_POLITICAL", "israel": "ISR_POLITICAL", "knesset": "ISR_POLITICAL",
        "mansour_abbas": "ISR_POLITICAL", "gallant": "ISR_POLITICAL", "gantz": "ISR_POLITICAL",
        "idf": "ISR_MILITARY", "mossad": "ISR_MILITARY", "shin bet": "ISR_MILITARY",
        "abbas": "PAL_POLITICAL", "pa": "PAL_POLITICAL", "palestine": "PAL_POLITICAL",
        "hamas": "PAL_ORG", "haniyeh": "PAL_ORG", "sinwar": "PAL_ORG", "pij": "PAL_ORG",
        "militants": "PAL_RESISTANCE", "qassam": "PAL_RESISTANCE", "gaza": "PAL_RESISTANCE",
        "usa": "INTL_ACTORS", "un": "INTL_ACTORS", "icj": "INTL_ACTORS", "biden": "INTL_ACTORS", "trump": "INTL_ACTORS",
        "iran": "INTL_ACTORS", "lebanon": "INTL_ACTORS", "turkey": "INTL_ACTORS", "syria": "INTL_ACTORS", "hezbollah": "INTL_ACTORS",
        "saudi arabia": "INTL_ACTORS", "qatar": "INTL_ACTORS", "egypt": "INTL_ACTORS",
        "civilians": "CIVILIANS", "hostages": "CIVILIANS", "settlers": "SETTLERS",
    }


class ClassifierConfig:
    """Shared config for the src/classifier/ pipeline (merge_data -> random_forest -> explainability/robustness)."""

    # label=0 = institutionally dominant side in each theater (Russian state media for
    # RU_UK; Israeli mainstream press for IL_PA). label=1 = challenger/resistance-side
    # outlets (Ukrainian press for RU_UK; Palestinian press for IL_PA).
    SOURCE_LABELS = {"kpru": 0, "rt": 0, "jpost": 0, "ynet": 0, "ynet_global": 0, "ukpravda": 1, "liganet": 1, "alquds": 1}

    THEATER_MAP = {
        "kpru": "RU_UK", "rt": "RU_UK", "ukpravda": "RU_UK", "liganet": "RU_UK",
        "jpost": "IL_PA", "ynet": "IL_PA", "ynet_global": "IL_PA", "alquds": "IL_PA",
    }

    SOURCE_INGROUP = {
        "kpru":        lambda l: "INGROUP"  if l.startswith("RU_")  else ("OUTGROUP" if l.startswith("UKR_") else "OTHER"),
        "rt":          lambda l: "INGROUP"  if l.startswith("RU_")  else ("OUTGROUP" if l.startswith("UKR_") else "OTHER"),
        "ukpravda":    lambda l: "INGROUP"  if l.startswith("UKR_") else ("OUTGROUP" if l.startswith("RU_")  else "OTHER"),
        "liganet":     lambda l: "INGROUP"  if l.startswith("UKR_") else ("OUTGROUP" if l.startswith("RU_")  else "OTHER"),
        "jpost":       lambda l: "INGROUP"  if l.startswith("ISR_") else ("OUTGROUP" if l.startswith("PAL_") else "OTHER"),
        "ynet":        lambda l: "INGROUP"  if l.startswith("ISR_") else ("OUTGROUP" if l.startswith("PAL_") else "OTHER"),
        "ynet_global": lambda l: "INGROUP"  if l.startswith("ISR_") else ("OUTGROUP" if l.startswith("PAL_") else "OTHER"),
        "alquds":      lambda l: "INGROUP"  if l.startswith("PAL_") else ("OUTGROUP" if l.startswith("ISR_") else "OTHER"),
    }

    MIN_ENTITY_HITS = 2
    MAX_CHUNKS_PER_SOURCE = 3000

    # Excluded from the primary (ablated) model -- see src/classifier/random_forest.py
    # for the rationale (both are global register signals, not structural features).
    REGISTER_FEATURES = ["toxicity_score", "chunk_sentiment"]

    RF_PARAMS = dict(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)

    MERGED_DATA_CSV           = "classification_data.csv"
    TEST_INDICES_FILE         = "test_indices.joblib"
    ABLATED_FEATURE_LIST_FILE = "ablated_feature_list.joblib"
    ABLATED_MODEL_FILE        = "narrative_rf_model_ablated.joblib"
    FULL_MODEL_FILE           = "narrative_rf_model.joblib"
    ABLATED_RESULTS_CSV       = "detailed_model_results.csv"
    FULL_RESULTS_CSV          = "detailed_model_results_full.csv"
    ABLATION_ROC_PNG          = "ablation_roc_comparison.png"


class AgencyConfig:
    """Shared config for src/agency/{extract,plot_violence,plot_bars}.py."""

    MIN_OCCURRENCES = 3   # minimum times an entity must appear to be included in the report

    # "we/our/ours" -> IN_GROUP framing; "they/them/their/theirs" -> OUT_GROUP framing.
    # "us" is intentionally excluded -- too ambiguous (fires on "told us", "among us").
    PRONOUN_GROUPS = {
        "we":     "IN_GROUP",
        "our":    "IN_GROUP",
        "ours":   "IN_GROUP",
        "they":   "OUT_GROUP",
        "them":   "OUT_GROUP",
        "their":  "OUT_GROUP",
        "theirs": "OUT_GROUP",
    }

    SUBJECT_OBJECT_DEPS = {"nsubj", "nsubjpass", "dobj", "obj"}

    # Minimum verb/label appearances before a summary row is plotted --
    # shared by plot_violence.py and plot_bars.py.
    MIN_N = 10

    AGENCY_THRESHOLD = 0.50
    VIOLENCE_THRESHOLD = 0.05

    LABEL_COLORS = {
        # Russia-Ukraine theater
        "RU_POLITICAL":   "#c0392b",   # red
        "RU_MILITARY":    "#e74c3c",   # lighter red
        "UKR_POLITICAL":  "#2980b9",   # blue
        "UKR_MILITARY":   "#3498db",   # lighter blue
        "WEST_ACTORS":    "#2c3e50",
        "INTL_ACTORS":    "#2c3e50",
        "CIVILIANS":      "#8e44ad",   # purple
        # Israel-Palestine theater
        "ISR_POLITICAL":  "#2980b9",   # blue
        "ISR_MILITARY":   "#3498db",   # lighter blue
        "PAL_POLITICAL":  "#27ae60",   # green
        "PAL_ORG":        "#2ecc71",   # lighter green
        "PAL_RESISTANCE": "#16a085",   # teal-green
        "SETTLERS":       "#f39c12",   # amber
    }
    DEFAULT_LABEL_COLOR = "#bdc3c7"

    # Colors encode outlet side (Russian/Ukrainian or Israeli/Palestinian) so the
    # reader can immediately read cross-side comparisons without a legend lookup.
    # Outlet display names come from Websites.DISPLAY_NAMES.
    THEATER_CONFIG = {
        "ru_ua": {
            "outlets": ["kpru", "rt", "ukpravda", "liganet"],
            # Red tones = Russian-side outlets; blue tones = Ukrainian-side outlets
            "outlet_colors": {
                "kpru":     "#c0392b",
                "rt":       "#e74c3c",
                "ukpravda": "#2980b9",
                "liganet":  "#3498db",
            },
            # Labels to display, in left-to-right order on the x-axis.
            # IN_GROUP and OUT_GROUP are excluded: they are pronoun-level labels
            # analyzed separately in §4.3.1 and would duplicate the entity-level
            # signal here.
            "labels": ["RU_POLITICAL", "UKR_POLITICAL", "WEST_ACTORS", "INTL_ACTORS", "CIVILIANS"],
            "agency_csv_pattern": "{outlet}_agency_actions.csv",
        },
        "il_pa": {
            "outlets": ["alquds", "jpost", "ynet", "ynet_global"],
            # Green tones = Palestinian-side; blue tones = Israeli-side
            "outlet_colors": {
                "alquds":      "#05714B",
                "jpost":       "#2980b9",
                "ynet":        "#3498db",
                "ynet_global": "#5dade2",
            },
            "labels": ["ISR_POLITICAL", "PAL_POLITICAL", "PAL_ORG", "PAL_RESISTANCE", "INTL_ACTORS", "CIVILIANS"],
            "agency_csv_pattern": "{outlet}_agency_actions.csv",
        },
    }


class CharactersConfig:
    """Shared config for src/characters/{extract,plot_adjectives}.py."""

    MIN_OCCURRENCES = 1   # minimum adjective rows per entity label to include in report
    # Set to 1 so low-frequency labels (CIVILIANS, SETTLERS) are not silently dropped.
    # Raise to 3-5 to exclude statistically thin labels from the summary.

    TOP_N = 8
    MORALITY_VMIN = -2.5
    MORALITY_VMAX = 2.5
    MORALITY_CMAP_COLORS = ["#c0392b", "#f5f5f5", "#27ae60"]

    # Adjectives with no character valence, filtered out of the top-N bars
    # (distinct from ADJ_STOPLIST below, which filters at extraction time).
    ADJ_STOPWORDS = {
        'more', 'less', 'entire', 'whole', 'about', 'pan', 'cross', 'primary',
        'vast', 'just', 'right', 'aware', 'alive', 'special', 'basic', 'mere',
        'such', 'own', 'very', 'certain', 'other',
    }

    # Temporal, ordinal, and generic adjectives that carry no character valence.
    # Projecting these onto competence/morality axes produces noise.
    ADJ_STOPLIST = {
        # --- Temporal / ordinal ---
        "recent", "new", "past", "current", "last", "first", "second", "third",
        "fourth", "fifth", "former", "late", "early", "annual", "daily",
        "next", "previous", "prior", "final", "initial", "ongoing", "constant",
        "pending", "upcoming", "scheduled", "planned", "active", "future",

        # --- Generic / scope ---
        "other", "same", "full", "able", "certain", "main", "overall", "general",
        "specific", "particular", "various", "several", "many", "few", "enough",
        "only", "clear", "direct", "open", "close", "hard", "heavy", "such",
        "actual", "sufficient", "difficult", "formal", "physical", "quiet",
        "different", "fundamental", "broad", "total", "comparable", "comparative",
        "bottom", "own", "possible", "complete", "critical", "positive",
        "multiple", "single", "available", "necessary", "relevant", "appropriate",
        "effective", "successful", "viable", "relative", "absolute", "universal",
        "extreme", "severe", "intense", "moderate",

        # --- Size / degree ---
        "old", "large", "small", "high", "low", "long", "short", "great",
        "big", "little", "wide", "narrow", "deep", "strong", "weak",

        # --- Colors (appear in "red line", "white phosphorus", "black market") ---
        "red", "blue", "green", "white", "black", "gray", "grey",
        "orange", "yellow", "purple", "brown", "dark", "light",

        # --- Demographic ---
        "young", "adult", "elderly", "female", "male",

        # --- Evaluative but content-free ---
        "good", "bad", "important", "major", "minor", "significant",
        "key", "central", "senior", "junior", "top", "leading", "historic",
        "chief",

        # --- Domain / relational (describe topic, not character) ---
        "military", "political", "diplomatic", "economic", "nuclear",
        "foreign", "regional", "international", "internal", "global",
        "national", "local", "federal", "official", "legal", "civil",
        "armed", "northern", "southern", "eastern", "western", "central",
        "strategic", "operational", "ballistic", "kinetic", "supreme",
        "scientific", "academic", "humanitarian", "ideological",
        "royal", "multinational", "intensive", "civilian", "covert",
        # Military domain
        "tactical", "conventional", "aerial", "naval", "ground",
        # Financial domain
        "financial", "fiscal", "monetary", "budgetary",
        # Religious/social domain
        "religious", "secular",
        # Medical domain
        "medical", "psychological", "psychiatric",
        # Administrative/legal domain
        "administrative", "governmental", "constitutional", "provisional", "interim",
        # Structural domain
        "structural", "systematic", "systemic", "institutional",
        # Diplomatic scope
        "bilateral", "unilateral", "multilateral",

        # --- Status descriptors ---
        "underway",

        # --- Scope / logical ---
        "potential", "partial", "immediate", "alternative", "comprehensive",
        "classic", "cognitive", "additional", "separate", "joint", "mutual",
        "common", "public", "private", "real", "true", "false",

        # --- Morphological prefixes spaCy tags as ADJ ---
        "anti", "pro", "post", "non", "pre", "neo", "ex",
    }


class CoOccurrenceConfig:
    """Shared config for src/co_occurrence/{extract,visualize}.py."""

    MIN_COUNT = 3   # minimum sentence co-occurrences to include a pair in output

    # Force-annotate entity pairs of narrative interest on the scatter plot,
    # curated per outlet from a prior read of each outlet's PMI/Jaccard results.
    HIGHLIGHT_PAIRS_UKPRAVDA = [
        ("yermak", "zelensky"),      # Ukrainian leadership dyad — high PMI (5.57), absent from Russian outlets
        ("iaea", "zaporizhzhia"),    # Nuclear risk frame — unique to Ukrainian coverage
        ("crimea", "donetsk"),       # Territorial frame — links occupied territories
        ("ukraine", "un"),           # Near-zero PMI despite 525 co-occurrences — ubiquitous but non-specific
    ]
    HIGHLIGHT_PAIRS_KPRU = [
        ("france", "germany"),    # Dominant outlier: Western bloc treated as unified actor (PMI=5.96, Jac=0.26)
        ("kremlin", "putin"),     # Institutional synecdoche — the building stands for the man
        ("azov", "russia"),       # Denazification frame — links Azov regiment to Russia's stated rationale
        ("putin", "trump"),       # High Jaccard (0.145) + meaningful PMI — most frequent diplomatic pair
        ("russia", "un"),         # Near-zero PMI despite 230 co-occurrences — Russia's UN omnipresence
    ]
    HIGHLIGHT_PAIRS_ALQUDS = [
        ("qatar", "turkey"),         # Highest PMI (6.25) AND highest Jaccard (0.27) simultaneously — regional mediator axis
        ("hezbollah", "lebanon"),    # Hezbollah framed as Lebanese political actor, not isolated armed group
        ("egypt", "qatar"),          # Mediation triangle: Egypt-Qatar-Turkey as diplomatic framework
        ("pa", "un"),                # Near-zero PMI despite 1170 co-occurrences — PA omnipresence, statistically expected
        ("israel", "un"),            # Near-zero PMI despite 624 count — Israel equally ubiquitous, no specific association
    ]
    HIGHLIGHT_PAIRS_JPOST = [
        ("palestine", "settlers"),   # PMI=6.10 — unique to J.Post across all outlets; ties settler identity to Palestinian context
        ("gaza", "hamas"),           # Gaza=Hamas equation: PMI=2.17, Jac=0.16 — substantive and frequent
        ("hezbollah", "lebanon"),    # Mirror of Al-Quds framing — worth noting the convergence
        ("iran", "israel"),          # PMI=-0.116, Jac=0.12 — NEGATIVE: Iran-Israel mentioned together less than chance despite 266 co-occurrences
        ("israel", "un"),            # Near-zero PMI despite 469 count — mirrors Al-Quds's israel-un finding exactly
    ]
    HIGHLIGHT_PAIRS_YNET = [
        ("qatar", "turkey"),         # Highest Jaccard across ALL Israel-Palestine outlets (0.386) + high PMI — stronger bloc framing than even Al-Quds
        ("abbas", "mansour_abbas"),  # Highest PMI (6.32) — Ynet tracks Palestinian leadership dyad with same specificity as Al-Quds; unexpected for an Israeli source
        ("biden", "idf"),            # US president linked specifically to Israeli military — accountability framing
        ("gaza", "palestine"),       # Gaza=Palestine equation from an Israeli outlet — territorial conflation
        ("israel", "un"),            # Near-zero PMI despite 271 co-occurrences — exact mirror of J.Post and Al-Quds finding
    ]
    HIGHLIGHT_PAIRS_RT = [
        ("france", "germany"),    # Dominant outlier: same Western-bloc homogenization as KP.RU (PMI=4.21, Jac=0.15) — cross-outlet convergence
        ("biden", "trump"),       # US presidents conflated as interchangeable — anti-Western framing regardless of administration
        ("usa", "yermak"),        # US linked to Ukrainian chief of staff — Ukraine-as-proxy narrative
        ("trump", "zelensky"),    # High Jaccard (0.078) + meaningful PMI — peace-deal negotiation frame
        ("russia", "zelensky"),   # Negative PMI (-0.544) despite 27 co-occurrences — RT systematically avoids linking Russia and Zelensky in the same context
    ]
    HIGHLIGHT_PAIRS_LIGANET = [
        ("yermak", "zelensky"),  # Highest PMI (3.74) + second-highest Jaccard — Ukrainian leadership dyad, mirrors Ukr. Pravda
        ("nato", "putin"),       # NATO framed as Putin's direct counterpart, not Ukraine's — agency attribution
        ("putin", "trump"),      # High Jaccard (0.133) + strong PMI — peace-deal negotiation frame, mirrors Liga.net's political focus
        ("ukraine", "un"),       # Near-zero PMI despite 81 co-occurrences — ubiquitous but non-specific
        ("ukraine", "usa"),      # Negative PMI (-0.414) despite 15 count — Liga.net avoids linking Ukraine and USA in shared contexts
    ]

    @classmethod
    def highlight_pairs_by_website(cls):
        return {
            "ukpravda":    cls.HIGHLIGHT_PAIRS_UKPRAVDA,
            "kpru":        cls.HIGHLIGHT_PAIRS_KPRU,
            "alquds":      cls.HIGHLIGHT_PAIRS_ALQUDS,
            "jpost":       cls.HIGHLIGHT_PAIRS_JPOST,
            "ynet":        cls.HIGHLIGHT_PAIRS_YNET,
            "ynet_global": cls.HIGHLIGHT_PAIRS_YNET,
            "rt":          cls.HIGHLIGHT_PAIRS_RT,
            "liganet":     cls.HIGHLIGHT_PAIRS_LIGANET,
        }


class NetworksConfig:
    """Shared config for src/networks/{extract,visualize}.py."""

    MEANINGFUL_PREPS = {"at", "against", "into", "on", "upon", "toward", "towards", "over"}


class SemanticDivergenceConfig:
    """Shared config for src/semantic_divergence/{compute,visualize}.py."""

    CONCEPTS_RU_UK = ["security", "peace", "justice", "war", "civilian", "russia", "ukraine", "putin", "zelensky", "nato"]
    CONCEPTS_IL_PA = ["security", "peace", "justice", "war", "civilian", "israel", "palestine", "hamas", "idf", "netanyahu"]

    # min_count=3 rather than 10: with small corpora (<100k tokens), min_count=10
    # aggressively prunes vocabulary to the point of producing degenerate embeddings
    # (as evidenced by centroid drift=0.0 and Jaccard>0.9 everywhere). min_count=3
    # gives ~5-10x more vocabulary types on typical news corpora. Log-odds uses its
    # own independent frequency threshold (prior smoothing), unaffected by this.
    W2V_MIN_COUNT = 3
    W2V_VOCAB_THRESHOLD = 2000   # below this, W2V geometry is unreliable


class PreprocessingConfig:
    """Shared config for src/preprocessing/*.py."""

    MAX_WORDS_PER_CHUNK = 250   # coreferencing: sub-chunk size fed to the coref model
    MAX_CHUNK_CHARS = 800       # chunking: target chunk length in characters
    FILTER_THRESHOLD = 0.4      # filtering: minimum topic-similarity score to keep a chunk
    TOXICITY_BATCH_SIZE = 32    # detoxify: batch size for toxicity scoring

    # Boilerplate footer some scraped sites append to article text; stripped
    # before chunking so it doesn't get split across chunks.
    STOP_MARKER = "The use of site materials is allowed only"


class EdaConfig:
    """Shared config for src/eda.py."""

    RU_UK_OUTLETS = {"KP.RU": "kpru", "RT": "rt", "Ukrainska Pravda": "ukpravda", "Liga.net": "liganet"}
    IL_PA_OUTLETS = {"Jerusalem Post": "jpost", "Ynet": "ynet", "Ynet Global": "ynet_global", "Al-Quds": "alquds"}

    SIDE_MAP_RU_UK = {"KP.RU": "Russian", "RT": "Russian", "Ukrainska Pravda": "Ukrainian", "Liga.net": "Ukrainian"}
    SIDE_MAP_IL_PA = {"Jerusalem Post": "Israeli", "Ynet": "Israeli", "Ynet Global": "Israeli", "Al-Quds": "Palestinian"}

    RU_UK_COLORS = {"Russian": "#C0392B", "Ukrainian": "#2980B9"}
    IL_PA_COLORS = {"Israeli": "#1A5276", "Palestinian": "#1E8449"}
