"""
Interactive CLI for the master-thesis analysis pipeline.

Replaces the old "one colab snippet per analysis, edit the hardcoded website
at the top, re-run the cell" workflow: every analysis script now exposes a
`main(...)` function (see src/preprocessing, src/agency, src/characters,
src/co_occurrence, src/networks, src/semantic_divergence, src/classifier,
src/eda), and this module lets the user pick which one to run and with what
parameters, entirely at runtime.

Each analysis module is imported lazily (only once the user actually selects
it), so launching the CLI and browsing the menu never pays the cost of
loading spaCy/transformers/gensim models for analyses the user didn't pick.
"""

import importlib

from src.utils.constants import Websites

THEATER_RU_UK = "Russia-Ukraine"
THEATER_IL_PA = "Israel-Palestine"


def _prompt(title, options, zero_label="Back"):
    """
    options: list of (label, value) tuples.
    Returns the selected value, or None if the user chose the "0" option.
    """
    while True:
        print(f"\n{title}")
        for i, (label, _value) in enumerate(options, start=1):
            print(f"  {i}. {label}")
        print(f"  0. {zero_label}")
        try:
            choice = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if choice == "0" or choice == "":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1][1]
        print("Invalid choice, try again.")


def _prompt_website(title="Which website?"):
    options = [(f"{w}  ({THEATER_RU_UK})", w) for w in Websites.WEBSITES_UKRAINE_RUSSIA]
    options += [(f"{w}  ({THEATER_IL_PA})", w) for w in Websites.WEBSITES_PALESTINE_ISRAEL]
    return _prompt(title, options)


def _prompt_theater(title="Which theater?"):
    options = [(THEATER_RU_UK, "ru_ua"), (THEATER_IL_PA, "il_pa")]
    return _prompt(title, options)


def _run(module_name, param_prompts=None, func_name="main"):
    """
    Lazily imports `module_name` and calls its `main(**kwargs)`, where kwargs
    is built by calling each prompt function in param_prompts. If any prompt
    returns None (user chose "Back"), the whole action is cancelled.
    """
    kwargs = {}
    for name, prompt_fn in (param_prompts or {}).items():
        value = prompt_fn()
        if value is None:
            print("Cancelled.")
            return
        kwargs[name] = value

    module = importlib.import_module(module_name)
    getattr(module, func_name)(**kwargs)


def _run_all_preprocessing():
    website = _prompt_website("Run the full preprocessing pipeline for which website?")
    if website is None:
        print("Cancelled.")
        return
    stages = [
        "src.preprocessing.translation_and_date",
        "src.preprocessing.coreferencing",
        "src.preprocessing.chunking",
        "src.preprocessing.embed",
        "src.preprocessing.filtering",
        "src.preprocessing.detoxify",
    ]
    for stage in stages:
        print(f"\n=== Running {stage} for {website} ===")
        importlib.import_module(stage).main(website)


def _run_full_classifier_pipeline():
    stages = [
        "src.classifier.merge_data",
        "src.classifier.random_forest",
        "src.classifier.explainability",
        "src.classifier.robustness",
    ]
    for stage in stages:
        print(f"\n=== Running {stage} ===")
        importlib.import_module(stage).main()


def _menu_preprocessing():
    actions = [
        ("Translate & clean dates", lambda: _run("src.preprocessing.translation_and_date", {"website": _prompt_website})),
        ("Coreference resolution", lambda: _run("src.preprocessing.coreferencing", {"website": _prompt_website})),
        ("Chunking", lambda: _run("src.preprocessing.chunking", {"website": _prompt_website})),
        ("Embeddings & topic filters", lambda: _run("src.preprocessing.embed", {"website": _prompt_website})),
        ("Similarity filtering", lambda: _run("src.preprocessing.filtering", {"website": _prompt_website})),
        ("Toxicity scoring (finalize)", lambda: _run("src.preprocessing.detoxify", {"website": _prompt_website})),
        ("Run all stages in sequence", _run_all_preprocessing),
    ]
    while True:
        action = _prompt("Preprocessing", actions)
        if action is None:
            return
        action()


def _menu_agency():
    actions = [
        ("Extract agency/violence data", lambda: _run("src.agency.extract", {"website": _prompt_website})),
        ("Plot agency-violence scatter", lambda: _run("src.agency.plot_violence", {"website": _prompt_website})),
        ("Plot agency bars (by theater)", lambda: _run("src.agency.plot_bars", {"theater": _prompt_theater})),
    ]
    while True:
        action = _prompt("Agency & polarization", actions)
        if action is None:
            return
        action()


def _menu_characters():
    actions = [
        ("Extract character/adjective data", lambda: _run("src.characters.extract", {"website": _prompt_website})),
        ("Plot adjective bars", lambda: _run("src.characters.plot_adjectives", {"website": _prompt_website})),
    ]
    while True:
        action = _prompt("Character framing", actions)
        if action is None:
            return
        action()


def _menu_co_occurrence():
    actions = [
        ("Extract co-occurrence pairs", lambda: _run("src.co_occurrence.extract", {"website": _prompt_website})),
        ("Visualize (heatmap + scatter)", lambda: _run("src.co_occurrence.visualize", {"website": _prompt_website})),
    ]
    while True:
        action = _prompt("Co-occurrence", actions)
        if action is None:
            return
        action()


def _menu_networks():
    actions = [
        ("Extract narrative network", lambda: _run("src.networks.extract", {"website": _prompt_website})),
        ("Visualize network (centrality/authority)", lambda: _run("src.networks.visualize", {"website": _prompt_website})),
    ]
    while True:
        action = _prompt("Networks", actions)
        if action is None:
            return
        action()


def _menu_semantic_divergence():
    actions = [
        ("Compute (train + save artifacts)", lambda: _run("src.semantic_divergence.compute", {
            "website1": lambda: _prompt_website("First outlet"),
            "website2": lambda: _prompt_website("Second outlet"),
        })),
        ("Visualize (regenerate plots)", lambda: _run("src.semantic_divergence.visualize", {
            "website1": lambda: _prompt_website("First outlet"),
            "website2": lambda: _prompt_website("Second outlet"),
        })),
    ]
    while True:
        action = _prompt("Semantic divergence", actions)
        if action is None:
            return
        action()


def _menu_classifier():
    actions = [
        ("Merge dataset (build classification_data.csv)", lambda: _run("src.classifier.merge_data")),
        ("Train random forest", lambda: _run("src.classifier.random_forest")),
        ("Explainability (SHAP, ROC, PDP)", lambda: _run("src.classifier.explainability")),
        ("Robustness (correlation, permutation importance)", lambda: _run("src.classifier.robustness")),
        ("Run full pipeline in sequence", _run_full_classifier_pipeline),
    ]
    while True:
        action = _prompt("Classifier", actions)
        if action is None:
            return
        action()


def _menu_eda():
    actions = [
        ("Run corpus statistics + plots", lambda: _run("src.eda")),
    ]
    while True:
        action = _prompt("EDA", actions)
        if action is None:
            return
        action()


def run():
    categories = [
        ("Preprocessing", _menu_preprocessing),
        ("Agency & polarization", _menu_agency),
        ("Character framing", _menu_characters),
        ("Co-occurrence", _menu_co_occurrence),
        ("Networks", _menu_networks),
        ("Semantic divergence", _menu_semantic_divergence),
        ("Classifier", _menu_classifier),
        ("EDA", _menu_eda),
    ]

    print("=" * 60)
    print("  Master Thesis — Narrative Analysis Pipeline")
    print("=" * 60)

    while True:
        category = _prompt("What do you want to run?", categories, zero_label="Exit")
        if category is None:
            print("Goodbye.")
            return
        category()


if __name__ == "__main__":
    run()
