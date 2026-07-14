import pandas as pd
import numpy as np
import spacy
import torch
import warnings
import logging
from tqdm import tqdm
from collections import Counter
from itertools import combinations
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from emfdscore.scoring import score_docs
from src.utils.constants import Axis, Verbs, NamesDicts, PretrainedModels



logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

class NarrativeAnalyzer:
    def __init__(self, website: str, device: str = None):
        self.website = website
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        self.nlp = spacy.load(PretrainedModels.SPACY_MODEL_LG)
        if not self.nlp.has_pipe("sentencizer"):
            self.nlp.add_pipe("sentencizer")
        
        self.analyzer = SentimentIntensityAnalyzer()
        
        try:
            self.coref_model = LingMessCoref(device=self.device)
        except:
            self.coref_model = FCoref(device=self.device)

        self._set_dictionaries()
        
        self.axis_seeds = Axis.AXIS_SEEDS
        self.comp_axis_vec = self._get_axis_vector(self.axis_seeds["competence"])
        self.moral_axis_vec = self._get_axis_vector(self.axis_seeds["morality"])

        self.VIOLENT_VERBS = Verbs.VIOLENT_VERBS
        self.MODAL_VERBS = Verbs.MODAL_VERBS
        self.HEDGES = Verbs.HEDGES

    def _set_dictionaries(self):
        """Consolidates the synonym and theater logic from Colab."""
        self.SYNONYM_MAP = NamesDicts.SYNONYM_MAP
        self.RU_UK_BASE = NamesDicts.RU_UK_BASE
        self.IZ_PA_BASE = NamesDicts.IZ_PA_BASE

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
    