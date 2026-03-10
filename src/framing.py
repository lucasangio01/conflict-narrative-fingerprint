import pandas as pd
import spacy
from tqdm import tqdm
from fastcoref import LingMessCoref
from collections import Counter
import re
from src.utils import constants
import torch



class Framing:
    def __init__(self, website):
        self.website = website
        self.df = pd.read_csv(f"{website}_toxicity.csv")
        texts = self.df['text'].astype(str).tolist()
        self.nlp = spacy.load("en_core_web_lg")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not self.nlp.has_pipe("sentencizer"):
            self.nlp.add_pipe("sentencizer")
        try:
            self.coref_model = LingMessCoref(device='cuda')
        except Exception as e:
            from fastcoref import FCoref
            self.coref_model = FCoref(device = 'cuda')

        self.RU_UK_BASE = {
            "russia": "RUSSIA", "russian": "RUSSIA", "moscow": "RUSSIA", "kremlin": "RUSSIA",
            "putin": "RUSSIA", "ukraine": "UKRAINE", "ukrainian": "UKRAINE", "kyiv": "UKRAINE",
            "zelensky": "UKRAINE", "civilians": "CIVILIANS", "refugees": "CIVILIANS",
            "army": "MILITARY", "troops": "MILITARY", "forces": "MILITARY"}

        self.IZ_PA_BASE = {
            "israel": "ISRAEL", "idf": "ISRAEL", "palestine": "PALESTINE", "hamas": "PALESTINE",
            "gaza": "PALESTINE", "civilians": "CIVILIANS", "settlers": "SETTLERS",
            "army": "MILITARY", "troops": "MILITARY", "forces": "MILITARY"}

        self.RU_UK_ALIASES = {"RUSSIA": ["russian army", "russian forces", "russian troops", "kremlin", "moscow", "russia", "russian", "putin"], "UKRAINE": ["ukrainian army", "ukrainian forces", "ukrainian troops", "kyiv", "kiev", "ukraine", "ukrainian", "zelensky"]}
        self.IZ_PA_ALIASES = {"ISRAEL": ["israeli army", "israeli forces", "idf", "tel aviv", "israel", "israeli", "netanyahu"], "PALESTINE": ["palestinian authority", "hamas", "gaza", "palestine", "palestinian", "abbas", "west bank"]}

        if self.website in ["ynet", "ynet_global", "alquds", "jpost"]:
            self.active_entities = self.IZ_PA_BASE
            self.active_aliases = self.IZ_PA_ALIASES
        else:
            self.active_entities = self.RU_UK_BASE
            self.active_aliases = self.RU_UK_ALIASES

        self.VIOLENT_VERBS = ["attack", "bomb", "kill", "destroy", "invade", "strike", "shell", "target", "raid"]
        self.MODAL_VERBS = {"may", "might", "could", "would", "should", "can"}
        self.HEDGES = {"seem", "appear", "likely", "possible", "allege", "probable", "suggest", "warn", "claim", "reportedly"}


    def get_group(self, token):
        text_to_check = token.text.lower()
        full_phrase = token.head.text.lower() if token.dep_ in ["compound", "amod"] else ""
        combined_text = f"{text_to_check} {full_phrase}".strip()

        for canonical, aliases in self.active_aliases.items():
            for alias in aliases:
                if re.search(rf'\b{alias}\b', text_to_check) or re.search(rf'\b{alias}\b', combined_text):
                    return canonical
        for key, group in self.active_entities.items():
            if re.search(rf'\b{key}\b', text_to_check):
                return group
        return None


    def calculate_uncertainty(self, sent):
        modals = sum(1 for token in sent if token.lemma_.lower() in self.MODAL_VERBS)
        hedges = sum(1 for token in sent if token.lemma_.lower() in self.HEDGES)
        return (modals + hedges) / len(sent) if len(sent) > 0 else 0


    def calculate_agency(self):
        all_actions = []
        for text in tqdm(self.texts, desc = "Analyzing Articles"):
            preds = self.coref_model.predict(texts=[text])
            res_obj = preds[0]
            if hasattr(res_obj, 'get_resolved_content'):
                resolved_text = res_obj.get_resolved_content()
            elif hasattr(res_obj, 'get_resolved_text'):
                resolved_text = res_obj.get_resolved_text()
            else:
                resolved_text = res_obj.resolved_text if hasattr(res_obj, 'resolved_text') else text
            doc = self.nlp(resolved_text)
            for sent in doc.sents:
                    sent_uncertainty = self.calculate_uncertainty(sent)
                    for token in sent:
                        canonical = self.get_group(token)
                        if canonical and token.head.pos_ == "VERB":
                            verb = token.head.lemma_.lower()
                            role = None
                            if token.dep_ == "nsubj": role = "AGENT"
                            elif token.dep_ in ["dobj", "nsubjpass"]: role = "PATIENT"
                            elif token.dep_ == "agent": role = "AGENT"
                            if role:
                                all_actions.append({"Entity": canonical, "Verb": verb, "Role": role, "is_violent": verb in self.VIOLENT_VERBS, "uncertainty": sent_uncertainty, "raw_sentence": sent.text})

        results_df = pd.DataFrame(all_actions)
        stats = results_df.groupby('Entity')['Role'].value_counts().unstack().fillna(0)
        if 'AGENT' not in stats: stats['AGENT'] = 0
        if 'PATIENT' not in stats: stats['PATIENT'] = 0

        stats['Agency_Ratio'] = stats['AGENT'] / (stats['AGENT'] + stats['PATIENT'])
        uncertainty_stats = results_df.groupby('Entity')['uncertainty'].mean()
        final_report = pd.concat([stats, uncertainty_stats], axis=1).sort_values("Agency_Ratio", ascending=False)

        print("\n--- Unified Agency & Uncertainty Analysis ---")
        print(final_report)
        results_df.to_csv(f"{self.website}_unified_analysis.csv", index=False)


    def adjectives_framing(self):
        adj_dict = {entity: [] for entity in self.entities}
        sent_count = {entity: 0 for entity in self.entities}

        for doc in self.nlp.pipe(self.texts, batch_size=50):
            for sent in doc.sents:
                sent_text = sent.text.lower()
                for ent in self.entities:
                    if ent.lower() in sent_text:
                        sent_count[ent] += 1
                        for token in sent:
                            if token.pos_ == "ADJ":
                                adj_dict[ent].append(token.lemma_.lower())

        rows = []
        for entity, adjs in adj_dict.items():
            counter = Counter(adjs)
            total_sents = sent_count[entity] if sent_count[entity] > 0 else 1
            for adj, count in counter.items():
                normalized = count / total_sents
                sentiment = self.sent_analyzer.polarity_scores(adj)["compound"]
                rows.append({"entity": entity, "adjective": adj, "count": count, "normalized_count": normalized, "sentiment": sentiment})

        df_adjectives = pd.DataFrame(rows)
        df_adjectives = df_adjectives.sort_values(["entity", "normalized_count"], ascending=[True, False])

        for ent in self.entities:
            print(f"\nTop adjectives around '{ent}':")
            print(df_adjectives[df_adjectives["entity"]==ent].head(10))
