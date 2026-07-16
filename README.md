# The Computational Narrative Fingerprint

Master's thesis in Data Science for Economics, Università degli Studi di Milano.
Author: Luca Sangiovanni. Supervisor: Prof. Alfio Ferrara. Co-supervisor: Prof. Silvia Salini.

This repository contains the full NLP pipeline behind the thesis *"The Computational
Narrative Fingerprint: A Structural NLP Analysis of Conflict Narratives in Online News
Media"* (`thesis.pdf`, presentation slides in `thesis_presentation.pdf`).

## What this is about

Most computational approaches to conflict journalism measure *what* a narrative says:
its topic, its sentiment, its polarity. This thesis measures *how* a narrative is built
instead. It introduces the Computational Narrative Fingerprint (CNF), a reproducible
pipeline that treats a news article as a structured system of agents, actions, moral
evaluations, and relational patterns, rather than a bag of words.

The framework draws on four theoretical traditions, Framing Theory, Agency-Patient
Theory, Moral Foundations Theory, and Social Identity Theory, and operationalizes each
computationally: who is grammatically cast as agent versus patient, how actors are
evaluated on competence and morality, how entities relate to each other across a
narrative network, and how shared vocabulary diverges in meaning between opposing
outlets.

It is applied to two geopolitically distinct conflicts:

- **Russia-Ukraine**: Komsomolskaya Pravda, Russia Today, Ukrainska Pravda, Liga.net
  (689 articles)
- **Israel-Palestine**: Jerusalem Post, Ynet, Ynet Global, Al-Quds (914 articles)

using a single theater-agnostic pipeline, so the same measurement architecture applies
to both without redesign.

## Research questions

1. Is agency distributed asymmetrically across conflict sides, and in what structural form?
2. Is the in-group moral advantage predicted by Social Identity Theory symmetric across
   opposing outlets within each theater?
3. Do opposing outlets construct divergent semantic spaces around shared conflict
   vocabulary?
4. Do these structural patterns generalize across two geopolitically distinct theaters?
5. Can a classifier trained only on structural features, with all named entities masked,
   recover which side of the conflict a text chunk comes from?

## What was found

Agency asymmetry, in-group moral elevation, and semantic divergence on normative terms
(above all "justice") show up in both theaters, but not identically: in Russia-Ukraine the
asymmetry is a bilateral mirror image, while in Israel-Palestine the structural role of
"agent" stays fixed on the same side regardless of which outlet is narrating, and only the
moral register inverts. A Random Forest trained exclusively on 19 structural features
extracted from entity-masked text recovers the narrative side above chance (AUC 0.708),
with agency asymmetry as the strongest predictor under both SHAP and permutation
importance.

## Pipeline and repository layout

The pipeline is organized as a sequence of stages, each a package under `src/`:

- `scrapers/` — async scrapers for the eight outlets
- `preprocessing/` — translation, date normalization, coreference resolution, chunking,
  topic-similarity filtering, toxicity scoring, and corpus-level EDA
- `agency/` — agent/patient role extraction and agency-violence analysis
- `characters/` — adjective-based competence/morality projection and Moral Foundations
  scoring
- `co_occurrence/` — entity co-occurrence, Jaccard and PMI association
- `networks/` — sentiment-weighted narrative networks (HITS centrality/authority)
- `semantic_divergence/` — log-odds lexical divergence and GloVe/Word2Vec neighborhood
  comparison
- `classifier/` — the structural-feature Random Forest, ablation study, and
  interpretability (SHAP, permutation importance)
- `utils/constants.py` — all shared configuration (entity dictionaries, thresholds, model
  identifiers, plotting config) in one place
- `cli.py` — an interactive menu that drives every stage above

Every stage exposes a `main(...)` function and can be run individually, in sequence, or
batched across outlets, all through the CLI.

## Running it

The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
uv run main.py
```

This launches the interactive CLI, which walks through scraping, preprocessing, and each
analysis module in turn, prompting for an outlet or outlet pair as needed.
