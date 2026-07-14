import functools
import re
import pandas as pd
import spacy
import torch
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel
from src.utils.constants import PretrainedModels, PreprocessingConfig


@functools.lru_cache(maxsize=1)
def get_nlp():
    """
    Lazily patches the transformers/fastcoref weight-tying quirks and loads
    the coref-augmented spaCy pipeline on first call, then caches it --
    importing this module for CLI listing purposes must not eagerly apply
    monkey-patches or load models.
    """
    def patched_check_and_enable_sdpa(cls, config, **kwargs):
        return config

    def patched_sdpa_can_dispatch(cls, **kwargs):
        return False

    PreTrainedModel._check_and_enable_sdpa = classmethod(patched_check_and_enable_sdpa)
    PreTrainedModel._sdpa_can_dispatch = classmethod(patched_sdpa_can_dispatch)

    try:
        import fastcoref.modeling as modeling
        for attr in dir(modeling):
            if "Coref" in attr:
                target_class = getattr(modeling, attr)
                if hasattr(target_class, "all_tied_weights_keys"):
                    target_class.all_tied_weights_keys = []
        print("✅ Successfully patched Coref weight-tying logic.")
    except Exception as e:
        print(f"⚠️ Patch warning: {e}")
    from fastcoref import spacy_component  # noqa: F401 -- registers the "fastcoref" spaCy factory

    nlp = spacy.load(PretrainedModels.SPACY_MODEL_SM, exclude=["parser", "lemmatizer", "ner", "textcat"])
    if "fastcoref" not in nlp.pipe_names:
        nlp.add_pipe("fastcoref", config={'model_architecture': 'LingMessCoref', 'model_path': 'biu-nlp/lingmess-coref', 'device': 'cuda'})
    return nlp


def run_full_article_coref_lingmess(df, website_name, nlp):
    df['text_en'] = df['text_en'].fillna("").astype(str)
    resolved_texts = []

    print(f"🧵 Processing {len(df)} articles for {website_name}...")

    for i, row in tqdm(df.iterrows(), total=len(df)):
        full_text = row['text_en']
        if not full_text.strip():
            resolved_texts.append("")
            continue

        sentences = re.split(r'(?<=[.!?])\s+', full_text)

        sub_chunks = []
        current_chunk = []
        current_word_count = 0

        for sent in sentences:
            words_in_sent = len(sent.split())
            if current_word_count + words_in_sent > PreprocessingConfig.MAX_WORDS_PER_CHUNK:
                if current_chunk:
                    sub_chunks.append(" ".join(current_chunk))
                current_chunk = [sent]
                current_word_count = words_in_sent
            else:
                current_chunk.append(sent)
                current_word_count += words_in_sent
        if current_chunk:
            sub_chunks.append(" ".join(current_chunk))

        resolved_sub_chunks = []
        try:
            for chunk in sub_chunks:
                doc = nlp(chunk, component_cfg={"fastcoref": {'resolve_text': True}})
                resolved_sub_chunks.append(doc._.resolved_text if doc._.resolved_text else chunk)

            resolved_texts.append(" ".join(resolved_sub_chunks))

        except Exception as e:
            print(f"\n⚠️ Row {i} failed. Error: {e}")
            resolved_texts.append(full_text)

        if i % 5 == 0:
            torch.cuda.empty_cache()

    df['resolved_text'] = resolved_texts
    output_file = f"3_{website_name}_resolved.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Mission Accomplished! Data saved to: {output_file}")
    return df


def main(website="alquds"):
    try:
        raw_df = pd.read_csv(f"2_{website}_english.csv", low_memory=False)
        df_resolved = run_full_article_coref_lingmess(raw_df, website, get_nlp())
    except Exception as e:
        print(f"❌ Execution Error: {e}")


if __name__ == "__main__":
    main()
