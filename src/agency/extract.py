import pandas as pd
import spacy
import warnings
import numpy as np
from tqdm import tqdm
from transformers import pipeline
from src.utils.constants import NamesDicts, Verbs, Websites, AgencyConfig, PretrainedModels

warnings.filterwarnings('ignore')


def main(website="alquds"):
    clean_data_file = f"{website}_final.csv"

    nlp = spacy.load(PretrainedModels.SPACY_MODEL_LG)

    print("✨ Initializing RoBERTa Sentiment Transformer...")
    try:
        sentiment_task = pipeline("sentiment-analysis", model=PretrainedModels.SENTIMENT_MODEL, device=0, batch_size=16)
        print("   ✅ Running on GPU")
    except Exception as e:
        print(f"   ⚠️  GPU unavailable ({e}), falling back to CPU")
        sentiment_task = pipeline("sentiment-analysis", model=PretrainedModels.SENTIMENT_MODEL, device=-1)

    is_il_pa = website in Websites.WEBSITES_PALESTINE_ISRAEL
    active_entities = NamesDicts.IZ_PA_BASE if is_il_pa else NamesDicts.RU_UK_BASE
    theater_name    = "Israel-Palestine" if is_il_pa else "Russia-Ukraine"

    search_keys = sorted(list(set(list(NamesDicts.SYNONYM_MAP.keys()) + list(active_entities.keys()))), key=len, reverse=True)

    print(f"🌍 Theater Detected: {theater_name} — website: {website}")

    def get_entity_from_token(token, doc):
        """
        Matches a token to a known entity using two strategies in order:
          1. Full noun chunk the token belongs to  (best for multi-word entities)
          2. ±3 token window fallback
        Returns (canonical_name, label) or (None, None).
        Only called on tokens in syntactic subject/object positions.
        """
        chunk_text = next((chunk.text.lower() for chunk in doc.noun_chunks if token.i in range(chunk.start, chunk.end)), None)
        window_text = " ".join(t.text.lower() for t in doc[max(0, token.i - 3): min(len(doc), token.i + 4)])

        for candidate_text in filter(None, [chunk_text, window_text]):
            match = next((k for k in search_keys if k in candidate_text), None)
            if match:
                clean = NamesDicts.SYNONYM_MAP.get(match, match)
                if clean in active_entities:
                    return clean, active_entities[clean]

        return None, None

    def resolve_token(token, doc):
        """
        Resolves a syntactic subject/object token to either:
          - A named entity (name, label) from the theater dictionary
          - A pronoun group (lemma, group_label) if the token is a tracked pronoun
            in a subject or object dependency position
        Returns (name, label, is_pronoun) or (None, None, False).
        """
        name, label = get_entity_from_token(token, doc)
        if name:
            return name, label, False

        if token.dep_ in AgencyConfig.SUBJECT_OBJECT_DEPS:
            lemma = token.lemma_.lower()
            if lemma in AgencyConfig.PRONOUN_GROUPS:
                return lemma, AgencyConfig.PRONOUN_GROUPS[lemma], True

        return None, None, False

    def calculate_uncertainty(sent):
        """
        Proportion of tokens in the sentence that are modal verbs or hedge markers.
        Returns float in [0, 1].
        """
        if len(sent) == 0:
            return 0.0
        count = sum(1 for t in sent if t.lemma_.lower() in Verbs.MODAL_VERBS or t.lemma_.lower() in Verbs.HEDGES)
        return round(count / len(sent), 4)

    def build_sentiment_cache(texts_list):
        """
        Collects all unique sentences across the corpus, scores them in one
        batched RoBERTa pass, and returns a dict {sentence_text: polarity}.
        Much faster than per-sentence inference inside the main loop, and
        consistent with the directed network script's approach.
        """
        print("📦 Collecting unique sentences for RoBERTa batch scoring...")
        unique_sents = set()
        for doc in nlp.pipe(texts_list, batch_size=32, disable=["ner"]):
            for sent in doc.sents:
                unique_sents.add(sent.text.strip())

        unique_sents = list(unique_sents)
        polarities = {}
        batch_size = 64

        print(f"🤖 Scoring {len(unique_sents):,} unique sentences with RoBERTa...")
        for i in tqdm(range(0, len(unique_sents), batch_size)):
            batch = unique_sents[i: i + batch_size]
            try:
                results = sentiment_task([s[:512] for s in batch])
                for sent_text, result in zip(batch, results):
                    label, score = result['label'], result['score']
                    if label == 'positive':   polarities[sent_text] = score
                    elif label == 'negative': polarities[sent_text] = -score
                    else:                     polarities[sent_text] = 0.0
            except Exception:
                for sent_text in batch:
                    polarities[sent_text] = 0.0

        return polarities

    df = pd.read_csv(clean_data_file)
    texts_list = df['text'].dropna().astype(str).tolist()
    all_actions = []

    sentiment_cache = build_sentiment_cache(texts_list)

    print(f"🚀 Analyzing {len(texts_list)} articles for {website}...")

    for doc in tqdm(nlp.pipe(texts_list, batch_size=32, disable=["ner"]), total=len(texts_list)):
        for sent in doc.sents:
            u_score = calculate_uncertainty(sent)
            s_score = sentiment_cache.get(sent.text.strip(), 0.0)

            for token in sent:

                if token.dep_ not in AgencyConfig.SUBJECT_OBJECT_DEPS:
                    continue

                if token.head.pos_ != "VERB":
                    continue

                ent_name, ent_label, is_pronoun = resolve_token(token, doc)
                if not ent_name:
                    continue

                verb = token.head.lemma_.lower()
                identity_group = (ent_label if is_pronoun else "NAMED_ENTITY")

                if token.dep_ == "nsubj":
                    all_actions.append({
                        "Entity":         ent_name,
                        "Label":          ent_label,
                        "Role":           "AGENT",
                        "Verb":           verb,
                        "Voice":          "active",
                        "Sentiment":      s_score,
                        "Uncertainty":    u_score,
                        "Is_Violent":     verb in Verbs.VIOLENT_VERBS,
                        "Is_Pronoun":     is_pronoun,
                        "Identity_Group": identity_group,
                    })

                elif token.dep_ in ("dobj", "obj"):
                    all_actions.append({
                        "Entity":         ent_name,
                        "Label":          ent_label,
                        "Role":           "PATIENT",
                        "Verb":           verb,
                        "Voice":          "active",
                        "Sentiment":      s_score,
                        "Uncertainty":    u_score,
                        "Is_Violent":     verb in Verbs.VIOLENT_VERBS,
                        "Is_Pronoun":     is_pronoun,
                        "Identity_Group": identity_group,
                    })

                elif token.dep_ == "nsubjpass":
                    all_actions.append({
                        "Entity":         ent_name,
                        "Label":          ent_label,
                        "Role":           "PATIENT",
                        "Verb":           verb,
                        "Voice":          "passive",
                        "Sentiment":      s_score,
                        "Uncertainty":    u_score,
                        "Is_Violent":     verb in Verbs.VIOLENT_VERBS,
                        "Is_Pronoun":     is_pronoun,
                        "Identity_Group": identity_group,
                    })

                    for child in token.head.children:
                        if child.dep_ == "agent":
                            for grandchild in child.children:
                                if grandchild.dep_ == "pobj":
                                    ag_name, ag_label, ag_pronoun = resolve_token(grandchild, doc)
                                    if ag_name:
                                        all_actions.append({
                                            "Entity":         ag_name,
                                            "Label":          ag_label,
                                            "Role":           "AGENT",
                                            "Verb":           verb,
                                            "Voice":          "passive",
                                            "Sentiment":      s_score,
                                            "Uncertainty":    u_score,
                                            "Is_Violent":     verb in Verbs.VIOLENT_VERBS,
                                            "Is_Pronoun":     ag_pronoun,
                                            "Identity_Group": ag_label if ag_pronoun else "NAMED_ENTITY",
                                        })

    results_df = pd.DataFrame(all_actions)

    if results_df.empty:
        print("⚠️  No actions extracted. Check your entity dictionary and input data.")
    else:
        entity_counts = results_df['Entity'].value_counts()
        valid_entities = entity_counts[entity_counts >= AgencyConfig.MIN_OCCURRENCES].index
        results_df = results_df[results_df['Entity'].isin(valid_entities)]

        label_agg = results_df.groupby('Label')

        label_report = (results_df.groupby(['Label', 'Role']).size().unstack(fill_value=0))
        for col in ['AGENT', 'PATIENT']:
            if col not in label_report.columns:
                label_report[col] = 0

        label_report['Agency_Ratio']  = (label_report['AGENT'] / (label_report['AGENT'] + label_report['PATIENT'])).round(4)
        label_report['Violence_Rate'] = label_agg['Is_Violent'].mean().round(4)
        label_report['Avg_Sentiment'] = label_agg['Sentiment'].mean().round(4)
        label_report['Avg_Uncertainty'] = label_agg['Uncertainty'].mean().round(4)

        entity_report = (results_df.groupby(['Entity', 'Label', 'Role']).size().unstack(fill_value=0))
        for col in ['AGENT', 'PATIENT']:
            if col not in entity_report.columns:
                entity_report[col] = 0

        entity_report['Agency_Ratio'] = (entity_report['AGENT'] / (entity_report['AGENT'] + entity_report['PATIENT'])).round(4)
        pronoun_df = results_df[results_df['Is_Pronoun'] == True]
        if not pronoun_df.empty:
            identity_report = (pronoun_df.groupby(['Identity_Group', 'Role']).size().unstack(fill_value=0))
            for col in ['AGENT', 'PATIENT']:
                if col not in identity_report.columns:
                    identity_report[col] = 0
            identity_report['Agency_Ratio'] = (identity_report['AGENT'] / (identity_report['AGENT'] + identity_report['PATIENT'])).round(4)
        else:
            identity_report = pd.DataFrame()

        violent_df = results_df[results_df['Is_Violent'] == True]
        violent_verb_report = (violent_df.groupby(['Label', 'Verb']).size().reset_index(name='count').sort_values('count', ascending=False))

        print(f"\n{'='*60}")
        print(f"  AGENCY ANALYSIS REPORT: {website} ({theater_name})")
        print(f"{'='*60}")

        print("\n--- [LEVEL 1] LABEL CATEGORY AGENCY ---")
        print(label_report.sort_values('Agency_Ratio', ascending=False).to_string())

        print("\n--- [LEVEL 2] ENTITY-LEVEL AGENCY ---")
        print(entity_report.sort_values('Agency_Ratio', ascending=False).to_string())

        if not identity_report.empty:
            print("\n--- [LEVEL 3] PRONOUN IN-GROUP / OUT-GROUP FRAMING ---")
            print(identity_report.to_string())

        print("\n--- [LEVEL 4] TOP VIOLENT VERB USAGE BY LABEL ---")
        print(violent_verb_report.head(20).to_string(index=False))

    results_df.to_csv(f"{website}_agency_actions.csv", index=False)
    print(f"\n✅ Saved: {website}_agency_actions.csv — {len(results_df):,} action records")

    return results_df


if __name__ == "__main__":
    main()
