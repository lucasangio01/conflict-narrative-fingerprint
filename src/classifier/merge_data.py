import re
import pandas as pd
import numpy as np
import spacy
import torch
import warnings
from tqdm import tqdm
from transformers import pipeline
from emfdscore.scoring import score_docs
from src.utils.constants import NamesDicts, Verbs, Lexicons, Axis, ClassifierConfig, PretrainedModels, PreprocessingConfig
from src.utils.logging_config import get_logger

logger = get_logger("CLASSIFIER")


def main():
    warnings.filterwarnings('ignore')

    logger.info("Loading models...")
    nlp = spacy.load(PretrainedModels.SPACY_MODEL_LG)
    sentiment_task = pipeline(
        "sentiment-analysis",
        model=PretrainedModels.SENTIMENT_MODEL,
        device=0 if torch.cuda.is_available() else -1,
    )

    def get_axis_vector(seeds):
        h = np.mean([nlp(w).vector for w in seeds["high"] if nlp(w).has_vector], axis=0)
        l = np.mean([nlp(w).vector for w in seeds["low"]  if nlp(w).has_vector], axis=0)
        return h - l

    comp_vec  = get_axis_vector(Axis.AXIS_SEEDS["competence"])
    moral_vec = get_axis_vector(Axis.AXIS_SEEDS["morality"])

    def axis_projection(text, vec):
        """Cosine similarity between the text vector and a semantic axis vector."""
        doc = nlp(text)
        if not doc.has_vector or doc.vector_norm == 0:
            return 0.0
        vec_norm = np.linalg.norm(vec)
        if vec_norm == 0:
            return 0.0
        return float(np.dot(doc.vector / doc.vector_norm, vec / vec_norm))

    def build_mask_pattern(search_keys):
        """
        Compile one regex pattern for all search keys (longest first).
        Word boundaries are applied only where the key begins/ends with a word
        character, so dotted abbreviations such as 'u.s.' are handled correctly.
        """
        parts = []
        for k in search_keys:
            esc = re.escape(k)
            lb  = r'\b' if re.match(r'\w', k[0])  else ''
            rb  = r'\b' if re.match(r'\w', k[-1]) else ''
            parts.append(lb + esc + rb)
        return re.compile('(?:' + '|'.join(parts) + ')', flags=re.IGNORECASE)

    def soft_mask_text(text, active_entities, pattern):
        """Replace entity surface forms with [LABEL] tags."""
        def replace_match(m):
            key       = m.group(0).lower()
            canonical = NamesDicts.SYNONYM_MAP.get(key, key)
            label     = active_entities.get(canonical)
            return f"[{label}]" if label else m.group(0)
        return pattern.sub(replace_match, text)

    def extract_structural_features(text, valid_labels, perspective_fn):
        doc = nlp(text)
        ingroup_agent = ingroup_patient = outgroup_agent = outgroup_patient = 0
        passive_count = total_verbs = violence_ingroup = violence_outgroup = entity_hit_count = 0
        negation_outgroup_sents = outgroup_sents_total = dehuman_hits = adj_total = 0
        sent_ingroup_idx, sent_outgroup_idx, uncertainty_scores = [], [], []
        adj_comp_in, adj_comp_out, adj_moral_in, adj_moral_out = [], [], [], []

        for s_idx, sent in enumerate(doc.sents):
            h_count = sum(
                1 for t in sent
                if t.lemma_.lower() in Verbs.MODAL_VERBS or t.lemma_.lower() in Verbs.HEDGES
            )
            uncertainty_scores.append(h_count / max(len(sent), 1))

            has_ig = has_og = False
            has_neg = any(
                t.dep_ == "neg" or t.lemma_.lower() in Verbs.NEGATION_TOKENS
                for t in sent
            )

            for token in sent:
                clean_token = token.text.replace("[", "").replace("]", "")
                if clean_token not in valid_labels:
                    continue

                entity_label = clean_token
                role         = perspective_fn(entity_label)
                entity_hit_count += 1

                if role == "INGROUP":    has_ig = True
                elif role == "OUTGROUP": has_og = True

                if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                    total_verbs += 1
                    verb = token.head.lemma_.lower()
                    if role == "INGROUP":
                        ingroup_agent += 1
                        if verb in Verbs.VIOLENT_VERBS: violence_ingroup += 1
                    elif role == "OUTGROUP":
                        outgroup_agent += 1
                        if verb in Verbs.VIOLENT_VERBS: violence_outgroup += 1
                elif token.dep_ in ("dobj", "nsubjpass") and token.head.pos_ == "VERB":
                    if token.dep_ == "nsubjpass": passive_count += 1
                    if role == "INGROUP":    ingroup_patient += 1
                    elif role == "OUTGROUP": outgroup_patient += 1

                for child in token.children:
                    if child.pos_ == "ADJ":
                        adj_total += 1
                        c_s = axis_projection(child.text, comp_vec)
                        m_s = axis_projection(child.text, moral_vec)
                        if role == "INGROUP":
                            adj_comp_in.append(c_s);  adj_moral_in.append(m_s)
                        elif role == "OUTGROUP":
                            adj_comp_out.append(c_s); adj_moral_out.append(m_s)
                        if child.lemma_.lower() in Lexicons.DEHUMAN_WORDS:
                            dehuman_hits += 1

            if has_ig: sent_ingroup_idx.append(s_idx)
            if has_og:
                sent_outgroup_idx.append(s_idx)
                outgroup_sents_total += 1
                if has_neg: negation_outgroup_sents += 1

        ig_ar = ingroup_agent  / max(ingroup_agent  + ingroup_patient,  1)
        og_ar = outgroup_agent / max(outgroup_agent + outgroup_patient, 1)

        return {
            "agency_asymmetry":       ig_ar - og_ar,
            "passive_voice_rate":     passive_count / max(total_verbs, 1),
            "violence_asymmetry":     (violence_outgroup / max(outgroup_agent, 1))
                                      - (violence_ingroup / max(ingroup_agent, 1)),
            "uncertainty_score":      np.mean(uncertainty_scores) if uncertainty_scores else 0.0,
            "negation_rate":          negation_outgroup_sents / max(outgroup_sents_total, 1),
            "competence_asymmetry":   np.mean(adj_comp_in or [0]) - np.mean(adj_comp_out or [0]),
            "morality_asymmetry":     np.mean(adj_moral_in or [0]) - np.mean(adj_moral_out or [0]),
            "dehumanization_index":   dehuman_hits / max(adj_total, 1),
            "ingroup_outgroup_ratio": len(sent_ingroup_idx) / max(len(sent_outgroup_idx), 1),
            "_sent_ingroup_idx":      sent_ingroup_idx,
            "_sent_outgroup_idx":     sent_outgroup_idx,
            "_sents":                 [s.text for s in doc.sents],
            "_entity_hit_count":      entity_hit_count,
        }

    all_rows = []

    for website, label in ClassifierConfig.SOURCE_LABELS.items():
        logger.info(f"[{website}] label={label}")
        try:
            df = pd.read_csv(PreprocessingConfig.STAGE_FINAL.format(website=website)).dropna(subset=["text"])
        except Exception:
            continue

        theater      = ClassifierConfig.THEATER_MAP[website]
        active_ents  = NamesDicts.IZ_PA_BASE if theater == "IL_PA" else NamesDicts.RU_UK_BASE
        perspective  = ClassifierConfig.SOURCE_INGROUP[website]
        valid_labels = set(active_ents.values())

        search_keys  = sorted(list(set(list(NamesDicts.SYNONYM_MAP.keys()) + list(active_ents.keys()))), key=len, reverse=True)
        mask_pattern = build_mask_pattern(search_keys)
        raw_texts    = df["text"].astype(str).tolist()

        logger.info("Masking entities...")
        masked_texts = [soft_mask_text(t, active_ents, mask_pattern) for t in tqdm(raw_texts)]

        logger.info("Structural extraction...")
        structural = [extract_structural_features(t, valid_labels, perspective) for t in tqdm(masked_texts)]

        # Filter BEFORE sampling so MAX_CHUNKS_PER_SOURCE applies to valid chunks only
        valid_idx = [i for i, f in enumerate(structural) if f["_entity_hit_count"] >= ClassifierConfig.MIN_ENTITY_HITS]
        logger.info(f"Valid after entity filter: {len(valid_idx)}/{len(raw_texts)}")

        if not valid_idx:
            continue

        if len(valid_idx) > ClassifierConfig.MAX_CHUNKS_PER_SOURCE:
            rng = np.random.default_rng(42)
            valid_idx = sorted(rng.choice(valid_idx, size=ClassifierConfig.MAX_CHUNKS_PER_SOURCE, replace=False).tolist())

        structural      = [structural[i]   for i in valid_idx]
        df_filtered     = df.iloc[valid_idx]
        masked_to_score = [masked_texts[i] for i in valid_idx]

        logger.info("Sentiment & MFT (on masked text)...")
        chunk_sents = [p['score'] if p['label'] == 'positive' else -p['score'] for p in sentiment_task([t[:512] for t in masked_to_score])]
        mft_df = score_docs(pd.DataFrame({"text": masked_to_score}), 'emfd', 'all', 'bow', 'sentiment', len(masked_to_score))

        for i, feat in enumerate(structural):
            feat.update({
                "label":           label,
                "source":          website,
                "theater":         theater,
                "chunk_sentiment": chunk_sents[i],
                "toxicity_score":  float(df_filtered.iloc[i].get("toxicity", 0.0)),
            })
            for col in mft_df.columns:
                if any(f in col for f in ['care', 'fair', 'loy', 'auth', 'sanc']):
                    feat[f"mft_{col}"] = mft_df.iloc[i][col]
            all_rows.append(feat)

    master_df = pd.DataFrame(all_rows).drop(columns=["_sent_ingroup_idx", "_sent_outgroup_idx", "_sents", "_entity_hit_count"])
    master_df.to_csv(ClassifierConfig.MERGED_DATA_CSV, index=False)
    logger.info(f"Done. Saved {len(master_df)} rows to {ClassifierConfig.MERGED_DATA_CSV}")

    return master_df


if __name__ == "__main__":
    main()
