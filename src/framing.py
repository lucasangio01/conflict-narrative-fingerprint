import pandas as pd
import spacy
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



class Framing:
    def __init__(self, website):
        self.website = website
        self.df = pd.read_csv(f"{self.website}_toxicity.csv")
        self.texts = self.df["text"].tolist()
        self.nlp = spacy.load("en_core_web_sm")
        self.sent_analyzer = SentimentIntensityAnalyzer()
        self.entities_1 = ["Israel", "Palestine", "Hamas", "IDF", "Netanyahu", "Abbas", "Gaza", "civilians","settlers"]
        self.entities_2 = ["Russia", "Ukraine", "Putin", "Zelensky", "Russian army", "Ukrainian army", "civilians"]
        self.entity_aliases_1 = {"Israel": ["Israel", "Israeli"], "Palestine": ["Palestine", "Palestinian"], "Hamas": ["Hamas"], "IDF": ["IDF", "Israeli army"], "Netanyahu": ["Netanyahu"], "Abbas": ["Abbas"], "Gaza": ["Gaza"], "civilians": ["civilians"], "settlers": ["settlers"]}
        self.entity_aliases_2 = {"Russia": ["Russia", "Russian"], "Ukraine": ["Ukraine", "Ukrainian"], "Putin": ["Putin", "Vladimir Putin"], "Zelensky": ["Zelensky", "Volodymyr Zelensky"], "Russian army": ["Russian army", "Russian forces", "Russian troops"], "Ukrainian army": ["Ukrainian army", "Ukrainian forces", "Ukrainian troops"], "civilians": ["civilians"]}
        
        if self.website in ["ynet", "ynet_global", "alquds"]:
            self.entities = self.entities_1
            self.entity_aliases = self.entity_aliases_1
        else:
            self.entities = self.entities_2
            self.entity_aliases = self.entity_aliases_2


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
            display(df_adjectives[df_adjectives["entity"]==ent].head(10))


    def verb_framing(self):
        subject_verbs = {e: [] for e in self.entities_of_interest}
        object_verbs = {e: [] for e in self.entities_of_interest}
        passive_verbs = {e: [] for e in self.entities_of_interest}
        mention_counts = {e: 0 for e in self.entities_of_interest}

        def match_entity(text):
            for canonical, aliases in self.entity_aliases.items():
                for alias in aliases:
                    if alias.lower() in text.lower():
                        return canonical
            return None

        for doc in self.nlp.pipe(self.texts, batch_size=10):

            last_entity = None
            for sent in doc.sents:
                for token in sent:

                    canonical = None
                    if token.ent_type_:
                        canonical = match_entity(token.text)
                        if canonical:
                            last_entity = canonical
                    elif token.pos_ == "PRON" and last_entity:
                        canonical = last_entity

                    if canonical:
                        mention_counts[canonical] += 1

                        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                            subject_verbs[canonical].append(token.head.lemma_)

                        if token.dep_ == "nsubjpass" and token.head.pos_ == "VERB":
                            passive_verbs[canonical].append(token.head.lemma_)

                        if token.dep_ in ["dobj", "obj"] and token.head.pos_ == "VERB":
                            object_verbs[canonical].append(token.head.lemma_)

        for ent in self.entities_of_interest:
            print(f"\n===== {ent} =====")
            print("Top ACTIVE verbs (subject):", Counter(subject_verbs[ent]).most_common(10))
            print("Top OBJECT verbs:", Counter(object_verbs[ent]).most_common(10))
            print("Top PASSIVE verbs:", Counter(passive_verbs[ent]).most_common(10))

        agency_scores = {}
        for ent in self.entities_of_interest:
            active = len(subject_verbs[ent])
            passive = len(passive_verbs[ent])
            obj_verbs = len(object_verbs[ent])
            total = mention_counts[ent] if mention_counts[ent] > 0 else 1
            agency_scores[ent] = {"agency_score": active / total, "victim_score": (passive + obj_verbs) / total}

        agency_df = pd.DataFrame(agency_scores).T
        print("\n--- Agency & Victim Scores ---")
        print(agency_df)