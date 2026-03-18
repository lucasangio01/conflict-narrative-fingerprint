import pandas as pd
import numpy as np
import spacy
import torch
import warnings
import logging
from tqdm import tqdm
from collections import Counter
from itertools import combinations
from fastcoref import LingMessCoref, FCoref
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from emfdscore.scoring import score_docs



logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

class NarrativeAnalyzer:
    def __init__(self, website: str, device: str = None):
        self.website = website
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.nlp = spacy.load("en_core_web_lg")
        if not self.nlp.has_pipe("sentencizer"):
            self.nlp.add_pipe("sentencizer")
        
        self.analyzer = SentimentIntensityAnalyzer()
        
        try:
            self.coref_model = LingMessCoref(device=self.device)
        except:
            self.coref_model = FCoref(device=self.device)

        self._set_dictionaries()
        
        self.axis_seeds = {
            "competence": {
                "high": ["powerful", "strategic", "efficient", "capable", "advanced", "strong", "organized"],
                "low": ["weak", "failing", "disorganized", "incompetent", "chaotic", "vulnerable", "ineffective"]
            },
            "morality": {
                "high": ["righteous", "just", "innocent", "heroic", "moral", "civilized", "peaceful"],
                "low": ["cruel", "evil", "terrorist", "murderous", "corrupt", "barbaric", "aggressive"]
            }
        }
        self.comp_axis_vec = self._get_axis_vector(self.axis_seeds["competence"])
        self.moral_axis_vec = self._get_axis_vector(self.axis_seeds["morality"])

        self.VIOLENT_VERBS = ["attack", "bomb", "kill", "destroy", "invade", "strike", "shell", "target", "raid", "fire"]
        self.MODAL_VERBS = {"may", "might", "could", "would", "should", "can"}
        self.HEDGES = {"seem", "appear", "likely", "possible", "allege", "probable", "suggest", "warn", "claim", "reportedly"}

    def _set_dictionaries(self):
        """Consolidates the synonym and theater logic from Colab."""
        self.SYNONYM_MAP = {
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

        self.RU_UK_BASE = {
            "russia": "RU_POLITICAL", "putin": "RU_POLITICAL", "moscow": "RU_POLITICAL", "kremlin": "RU_POLITICAL",
            "ukraine": "UKR_POLITICAL", "zelensky": "UKR_POLITICAL", "kiev": "UKR_POLITICAL", "yermak": "UKR_POLITICAL",
            "zaporizhzhia": "UKR_POLITICAL", "crimea": "UKR_POLITICAL", "kharkiv": "UKR_POLITICAL", "donetsk": "UKR_POLITICAL",
            "usa": "WEST_ACTORS", "nato": "WEST_ACTORS", "eu": "WEST_ACTORS", "biden": "WEST_ACTORS", "trump": "WEST_ACTORS",
            "poland": "WEST_ACTORS", "germany": "WEST_ACTORS", "france": "WEST_ACTORS",
            "army": "RU_MILITARY", "wagner": "RU_MILITARY", "afu": "UKR_MILITARY", "azov": "UKR_MILITARY",
            "china": "INTL_ACTORS", "un": "INTL_ACTORS", "iaea": "INTL_ACTORS", "turkey": "INTL_ACTORS",
            "civilians": "CIVILIANS"
        }

        self.IZ_PA_BASE = {
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

        if self.website in ["ynet", "ynet_global", "alquds", "jpost", "aljazeera"]:
            self.active_entities = self.IZ_PA_BASE
        else:
            self.active_entities = self.RU_UK_BASE
        
        self.target_labels = list(set(self.active_entities.values()))
        self.search_keys = sorted(list(set(list(self.SYNONYM_MAP.keys()) + list(self.active_entities.keys()))), key=len, reverse=True)


    def _get_axis_vector(self, seeds):
        h = np.mean([self.nlp(w).vector for w in seeds["high"] if self.nlp(w).has_vector], axis=0)
        l = np.mean([self.nlp(w).vector for w in seeds["low"] if self.nlp(w).has_vector], axis=0)
        return h - l


    def _get_projection_score(self, adj_text, axis_vec):
        doc = self.nlp(adj_text)
        if not doc.has_vector or doc.vector_norm == 0: return 0.0
        return np.dot(doc.vector, axis_vec) / np.linalg.norm(axis_vec)


    def _get_canonical_label(self, token_text):
        clean_name = self.SYNONYM_MAP.get(token_text.lower(), token_text.lower())
        return self.active_entities.get(clean_name), clean_name


    def process_dataset(self, df):
        """Main execution loop covering all analyses."""
        texts = df['text'].dropna().astype(str).tolist()
        
        adj_data, agency_data, cooc_data = [], [], []
        mft_texts = {l: [] for l in self.target_labels}
        entity_freq = Counter()
        pair_counts = Counter()

        for text in tqdm(texts, desc=f"Analyzing {self.website}"):
            preds = self.coref_model.predict(texts=[text])
            resolved_text = preds[0].get_resolved_content() if hasattr(preds[0], 'get_resolved_content') else text
            doc = self.nlp(resolved_text)
            
            found_in_art = set()

            for sent in doc.sents:
                u_score = sum(1 for t in sent if t.lemma_.lower() in self.MODAL_VERBS | self.HEDGES) / (len(sent) or 1)
                
                for token in sent:
                    label, canonical = self._get_canonical_label(token.text)
                    if not label: continue
                    
                    found_in_art.add(canonical)
                    mft_texts[label].append(sent.text)

                    role, verb_token = None, None
                    if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                        role, verb_token = "AGENT", token.head
                    elif token.dep_ in ["nsubjpass", "dobj"] and token.head.pos_ == "VERB":
                        role, verb_token = "PATIENT", token.head
                    
                    if role:
                        agency_data.append({"Entity": label, "Role": role, "Verb": verb_token.lemma_.lower(), "uncertainty": u_score, "is_violent": verb_token.lemma_.lower() in self.VIOLENT_VERBS})

                    if token.dep_ in ["nsubj", "nsubjpass"]:
                        for child in token.head.children:
                            if child.dep_ in ["acomp", "amod"] and child.pos_ == "ADJ":
                                adj_data.append({"label": label, "adj": child.lemma_.lower(), "is_neg": any(c.dep_ == "neg" for c in child.children)})

            for ent in found_in_art: entity_freq[ent] += 1
            if len(found_in_art) >= 2:
                for pair in combinations(sorted(list(found_in_art)), 2):
                    pair_counts[pair] += 1

        return self._summarize(adj_data, agency_data, mft_texts, entity_freq, pair_counts)


    def _summarize(self, adj_data, agency_data, mft_texts, entity_freq, pair_counts):
        """Aggregation and DataFrame generation."""
        df_agency = pd.DataFrame(agency_data)
        agency_report = df_agency.groupby('Entity')['Role'].value_counts().unstack().fillna(0)
        agency_report['Agency_Ratio'] = agency_report['AGENT'] / (agency_report['AGENT'] + agency_report['PATIENT'])
        
        mft_results = []
        for label, texts in mft_texts.items():
            if texts:
                m_scores = score_docs(pd.DataFrame(texts, columns=['text']), 'emfd', 'all', 'bow', 'sentiment', len(texts))
                avg = m_scores.mean(numeric_only=True)
                avg['label'] = label
                mft_results.append(avg)
        
        return agency_report, pd.DataFrame(mft_results), pd.DataFrame(pair_counts.items())
    