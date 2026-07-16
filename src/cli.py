"""
Interactive CLI for the master-thesis analysis pipeline.

Replaces the old "one colab snippet per analysis, edit the hardcoded website
at the top, re-run the cell" workflow: every analysis script now exposes a
`main(...)` function (see src/scrapers, src/preprocessing, src/agency,
src/characters, src/co_occurrence, src/networks, src/semantic_divergence,
src/classifier), and this module lets the user pick which one to run and
with what parameters, entirely at runtime.

Each analysis module is imported lazily (only once the user actually selects
it), so launching the CLI and browsing the menu never pays the cost of
loading spaCy/transformers/gensim models for analyses the user didn't pick.
"""

import asyncio
import importlib
import inspect

from src.utils.constants import Websites
from src.utils.logging_config import get_logger

logger = get_logger("CLI")


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
    options = [(f"{w}  ({Websites.THEATER_RU_UK})", w) for w in Websites.WEBSITES_UKRAINE_RUSSIA]
    options += [(f"{w}  ({Websites.THEATER_IL_PA})", w) for w in Websites.WEBSITES_PALESTINE_ISRAEL]
    return _prompt(title, options)


def _prompt_theater(title="Which theater?"):
    options = [(Websites.THEATER_RU_UK, "ru_ua"), (Websites.THEATER_IL_PA, "il_pa")]
    return _prompt(title, options)


def _prompt_batch_scope(title="Run for which outlets?"):
    """Returns a list of website ids, or None if the user backed out."""
    options = [
        (f"All {Websites.THEATER_RU_UK} outlets", list(Websites.WEBSITES_UKRAINE_RUSSIA)),
        (f"All {Websites.THEATER_IL_PA} outlets", list(Websites.WEBSITES_PALESTINE_ISRAEL)),
        ("All outlets (both theaters)", list(Websites.WEBSITES_UKRAINE_RUSSIA) + list(Websites.WEBSITES_PALESTINE_ISRAEL)),
    ]
    return _prompt(title, options)


def _run(module_name, param_prompts=None, func_name="main"):
    """
    Lazily imports `module_name` and calls its `main(**kwargs)`, where kwargs
    is built by calling each prompt function in param_prompts. If any prompt
    returns None (user chose "Back"), the whole action is cancelled.
    `main` may be sync (every analysis module) or async (the scrapers, which
    are built on aiohttp) -- either is run to completion transparently.
    """
    kwargs = {}
    for name, prompt_fn in (param_prompts or {}).items():
        value = prompt_fn()
        if value is None:
            logger.info("Cancelled.")
            return
        kwargs[name] = value

    module = importlib.import_module(module_name)
    result = getattr(module, func_name)(**kwargs)
    if inspect.iscoroutine(result):
        asyncio.run(result)


def _run_batch(module_name, func_name="main"):
    """
    Prompts for an outlet scope (one theater, or all outlets), then calls
    module_name.main(website=w) for each outlet in turn. A failure on one
    outlet is reported and skipped rather than aborting the rest of the batch.
    """
    websites = _prompt_batch_scope()
    if websites is None:
        logger.info("Cancelled.")
        return

    module = importlib.import_module(module_name)
    for website in websites:
        logger.info(f"Running {module_name} for {website}")
        try:
            getattr(module, func_name)(website=website)
        except Exception as e:
            logger.error(f"{website} failed: {e}")


def _menu_batch(actions, title="Which action?"):
    """
    Generic "run for multiple outlets" sub-menu: lets the user pick which
    single-website action to batch, then delegates to _run_batch.
    `actions` is a list of (label, module_name) pairs.
    """
    module_name = _prompt(title, actions)
    if module_name is None:
        return
    _run_batch(module_name)


_SCRAPERS = [
    ("Jerusalem Post", "src.scrapers.israel.scraper_jpost"),
    ("Ynet", "src.scrapers.israel.scraper_ynet"),
    ("Ynet Global", "src.scrapers.israel.scraper_ynet_global"),
    ("Al-Quds", "src.scrapers.palestine.scraper_alquds"),
    ("Komsomolskaya Pravda (KP.RU)", "src.scrapers.russia.scraper_rt_kpru"),
    ("Liga.net", "src.scrapers.ukraine.scraper_liganet"),
    ("Ukrainska Pravda", "src.scrapers.ukraine.scraper_ukpravda"),
    # RT has no scraper in this codebase yet (see scrapers_russia.py's Russia
    # class docstring) -- deliberately omitted here rather than wired to
    # something broken.
]


def _run_all_scrapers():
    for label, module_name in _SCRAPERS:
        logger.info(f"Scraping {label}")
        try:
            _run(module_name)
        except Exception as e:
            logger.error(f"{label} failed: {e}")


def _menu_scraping():
    actions = [(label, (lambda m=module_name: _run(m))) for label, module_name in _SCRAPERS]
    actions.append(("Scrape all outlets in sequence", _run_all_scrapers))
    while True:
        action = _prompt("Scraping", actions)
        if action is None:
            return
        action()


def _run_all_preprocessing(website):
    stages = [
        "src.preprocessing.translation_and_date",
        "src.preprocessing.coreferencing",
        "src.preprocessing.chunking",
        "src.preprocessing.embed",
        "src.preprocessing.filtering",
        "src.preprocessing.detoxify",
    ]
    for stage in stages:
        logger.info(f"Running {stage} for {website}")
        try:
            importlib.import_module(stage).main(website)
        except Exception as e:
            logger.error(f"{stage} failed for {website}: {e}")
            return


def _run_all_preprocessing_single():
    website = _prompt_website("Run the full preprocessing pipeline for which website?")
    if website is None:
        logger.info("Cancelled.")
        return
    _run_all_preprocessing(website)


def _run_all_preprocessing_batch():
    websites = _prompt_batch_scope("Run the full preprocessing pipeline for which outlets?")
    if websites is None:
        logger.info("Cancelled.")
        return
    for website in websites:
        logger.info(f"{'#' * 60}\n# {website}\n{'#' * 60}")
        _run_all_preprocessing(website)


def _run_full_classifier_pipeline():
    stages = [
        "src.classifier.merge_data",
        "src.classifier.random_forest",
        "src.classifier.explainability",
        "src.classifier.robustness",
    ]
    for stage in stages:
        logger.info(f"Running {stage}")
        importlib.import_module(stage).main()


def _menu_preprocessing():
    actions = [
        ("Translate & clean dates", lambda: _run("src.preprocessing.translation_and_date", {"website": _prompt_website})),
        ("Coreference resolution", lambda: _run("src.preprocessing.coreferencing", {"website": _prompt_website})),
        ("Chunking", lambda: _run("src.preprocessing.chunking", {"website": _prompt_website})),
        ("Embeddings & topic filters", lambda: _run("src.preprocessing.embed", {"website": _prompt_website})),
        ("Similarity filtering", lambda: _run("src.preprocessing.filtering", {"website": _prompt_website})),
        ("Toxicity scoring (finalize)", lambda: _run("src.preprocessing.detoxify", {"website": _prompt_website})),
        ("Run all stages in sequence", _run_all_preprocessing_single),
        ("Run one stage for multiple outlets", lambda: _menu_batch([
            ("Translate & clean dates", "src.preprocessing.translation_and_date"),
            ("Coreference resolution", "src.preprocessing.coreferencing"),
            ("Chunking", "src.preprocessing.chunking"),
            ("Embeddings & topic filters", "src.preprocessing.embed"),
            ("Similarity filtering", "src.preprocessing.filtering"),
            ("Toxicity scoring (finalize)", "src.preprocessing.detoxify"),
        ], "Which stage?")),
        ("Run all stages for multiple outlets", _run_all_preprocessing_batch),
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
        ("Run for multiple outlets", lambda: _menu_batch([
            ("Extract agency/violence data", "src.agency.extract"),
            ("Plot agency-violence scatter", "src.agency.plot_violence"),
        ])),
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
        ("Run for multiple outlets", lambda: _menu_batch([
            ("Extract character/adjective data", "src.characters.extract"),
            ("Plot adjective bars", "src.characters.plot_adjectives"),
        ])),
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
        ("Run for multiple outlets", lambda: _menu_batch([
            ("Extract co-occurrence pairs", "src.co_occurrence.extract"),
            ("Visualize (heatmap + scatter)", "src.co_occurrence.visualize"),
        ])),
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
        ("Run for multiple outlets", lambda: _menu_batch([
            ("Extract narrative network", "src.networks.extract"),
            ("Visualize network (centrality/authority)", "src.networks.visualize"),
        ])),
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
        ("Run corpus statistics + plots", lambda: _run("src.preprocessing.eda")),
    ]
    while True:
        action = _prompt("EDA", actions)
        if action is None:
            return
        action()


def run():
    categories = [
        ("Scraping", _menu_scraping),
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
