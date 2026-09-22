import numpy as np
import pytest

from evotrader.evolution import Dataset, EvolutionConfig, Evolver
from evotrader.evolution import history as h
from tests.conftest import make_bars

CONFIG = EvolutionConfig(population=16, generations=6, elite=2, seed=2)


@pytest.fixture(scope="module")
def run():
    datasets = [Dataset(f"T{i}", make_bars(n=900, seed=i)) for i in range(3)]
    log = h.record_run(datasets, "2015-06-01", "2017-12-31", objective="sharpe", config=CONFIG)
    return log, datasets


@pytest.fixture(scope="module")
def df(run, tmp_path_factory):
    path = tmp_path_factory.mktemp("evo") / "log.json"
    h.save(run[0], path)
    return h.load(path)[1]


def test_log_has_every_individual(df):
    assert len(df) == CONFIG.population * CONFIG.generations
    assert df["id"].is_unique
    assert set(df["group"]) <= set(h.BIRTH_GROUPS)


def test_parents_come_from_the_previous_generation(df):
    gen_of = df.set_index("id")["generation"]
    for _, row in df[df["generation"] > 0].iterrows():
        assert row["parents"] or row["origin"] == "immigrant"
        for p in row["parents"]:
            assert gen_of[p] == row["generation"] - 1


def test_survivors_are_unchanged_elites(df):
    by_id = df.set_index("id")
    survivors = df[df["origin"] == "survivor"]
    assert len(survivors) == CONFIG.elite * (CONFIG.generations - 1)
    for _, s in survivors.iterrows():
        parent = by_id.loc[s["parents"][0]]
        assert parent["rule"] == s["rule"] and parent["rank"] < CONFIG.elite


def test_recording_does_not_change_the_search(run):
    """Same seed with and without the recorder must evolve identical populations."""
    _, datasets = run
    from evotrader.evolution import FitnessConfig
    kwargs = dict(config=CONFIG, fitness_config=FitnessConfig(objective="sharpe"))
    plain = Evolver(datasets, "2015-06-01", "2017-12-31", **kwargs).run()
    recorded = Evolver(datasets, "2015-06-01", "2017-12-31", **kwargs, record=True).run()
    assert [s.best_rule for s in plain.history] == [s.best_rule for s in recorded.history]
    assert not plain.log and recorded.log


def test_champion_lineage_is_unbroken(df):
    champ = h.champion(df)
    line = h.main_line(df, champ["id"])
    # One ancestor per generation back to its founder: a random, seeded or immigrant rule.
    gens = line["generation"].tolist()
    assert gens == list(range(gens[0], CONFIG.generations))
    assert line.iloc[0]["origin"] in ("random", "seed", "immigrant")
    assert line.iloc[-1]["rule"] == champ["rule"]
    tree = h.ancestry(df, champ["id"])
    assert set(line["id"]) <= set(tree["id"])
    assert all(not p for p in tree.loc[tree["origin"].isin(["random", "immigrant"]), "parents"])


def test_story_collapses_unchanged_survival(df):
    s = h.story(df, h.champion(df)["id"])
    assert (s["rule"] != s["rule"].shift()).all()
    assert s["generation"].is_monotonic_increasing


def test_gene_pool_is_a_share(df):
    pool = h.gene_pool(df)
    assert ((pool >= 0) & (pool <= 1)).all().all()
    assert list(pool.columns) == list(range(CONFIG.generations))


def test_rule_tree_draws_every_condition(df):
    champ = h.champion(df)
    text = h.rule_tree(champ["genome"])
    assert text.startswith("ENTRY") and "EXIT" in text
    n_conditions = len(champ["kinds"])
    leaves = [line.split("-- ", 1)[1] for line in text.splitlines() if "-- " in line]
    assert sum(" < " in leaf or " > " in leaf for leaf in leaves) == n_conditions


def test_origins_describe_what_happened(df):
    origins = set(df["origin"])
    assert {"random", "survivor"} <= origins
    assert any(o.startswith("crossover") for o in origins)
    assert any(o.split(" (")[0] in {"threshold nudged", "window changed", "comparison flipped",
                                    "condition added", "condition replaced", "branch pruned",
                                    "subtree replaced", "mutation rejected"} for o in origins)
    assert np.isfinite(df["fitness"]).all()
