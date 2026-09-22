"""Evolution logs: record a GA run, save it, and turn it into things worth looking at.

A log is one row per individual per generation: its id, how it was born, its
parent ids, its rule and its *training* fitness. From that we can reconstruct:

- the population swarm    where every strategy sat, generation by generation
- the family tree         every ancestor of the final champion
- the main line           the champion's direct line of descent, step by step
- the gene pool           which indicators the population relies on over time
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from evotrader.evolution.engine import EvolutionConfig, Evolver
from evotrader.evolution.fitness import Dataset, FitnessConfig, truncate
from evotrader.evolution.genome import Compare, Genome, Node

# How individuals are grouped for colour: three groups, so identity stays readable.
BIRTH_GROUPS = ("offspring", "survivor", "newcomer")


def birth_group(origin: str) -> str:
    if origin == "survivor":
        return "survivor"
    if origin in ("random", "seed", "immigrant"):
        return "newcomer"
    return "offspring"


def record_run(
    datasets: list[Dataset], start: str, end: str, objective: str = "excess",
    config: EvolutionConfig | None = None, space: str = "timing",
) -> dict[str, Any]:
    """Run one evolution with recording on and return a JSON-ready log."""
    data = truncate(datasets, end)  # the run physically cannot see past `end`
    config = config or EvolutionConfig()
    evolver = Evolver(data, start, end, config, FitnessConfig(objective=objective),
                      space=space, record=True)
    result = evolver.run()
    return {
        "meta": {"objective": objective, "space": space, "start": start, "end": end,
                 "tickers": [ds.ticker for ds in datasets], **asdict(config)},
        "individuals": result.log,
    }


def save(log: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(log, separators=(",", ":")), encoding="utf-8")


def load(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(raw["individuals"])
    df["group"] = df["origin"].map(birth_group)
    return raw["meta"], df


def champion(df: pd.DataFrame) -> pd.Series:
    """The best individual of the final generation."""
    last = df[df["generation"] == df["generation"].max()]
    return last.sort_values("rank").iloc[0]


def ancestry(df: pd.DataFrame, individual_id: int) -> pd.DataFrame:
    """Every ancestor of an individual (both parents of every crossover), plus itself."""
    by_id = df.set_index("id")
    seen, stack = set(), [individual_id]
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        stack.extend(by_id.at[i, "parents"])
    return df[df["id"].isin(seen)].sort_values(["generation", "rank"])


def main_line(df: pd.DataFrame, individual_id: int) -> pd.DataFrame:
    """The direct line of descent: always follow the first parent (the one that was
    mutated, or the one that received the crossover graft)."""
    by_id = df.set_index("id")
    chain, i = [], individual_id
    while True:
        chain.append(i)
        parents = by_id.at[i, "parents"]
        if not parents:
            break
        i = parents[0]
    return df.set_index("id").loc[chain[::-1]].reset_index()


def story(df: pd.DataFrame, individual_id: int) -> pd.DataFrame:
    """The main line with unchanged survivals collapsed: one row per actual change."""
    line = main_line(df, individual_id)
    rows = []
    for _, r in line.iterrows():
        if rows and r["rule"] == rows[-1]["rule"]:
            rows[-1]["survived_until"] = int(r["generation"])
            continue
        rows.append({"generation": int(r["generation"]), "how": r["origin"],
                     "fitness": r["fitness"], "rule": r["rule"],
                     "survived_until": int(r["generation"])})
    out = pd.DataFrame(rows)
    out["change"] = out["fitness"].diff()
    return out


def gene_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Share of the population using each indicator, per generation (kinds x generations)."""
    uses = df[["generation", "kinds"]].copy()
    uses["kinds"] = uses["kinds"].map(lambda ks: sorted(set(ks)))
    exploded = uses.explode("kinds").dropna()
    counts = exploded.groupby(["kinds", "generation"]).size().unstack(fill_value=0)
    return counts / df.groupby("generation").size()


def rule_tree(genome_dict: dict[str, Any]) -> str:
    """Draw a genome's entry and exit rules as indented trees."""
    if genome_dict.get("type") == "rotation":
        terms = "\n".join(f"  {t['weight']:+.2f} x {t['feature']}"
                          f"({','.join(map(str, t['windows']))})" for t in genome_dict["terms"])
        return (f"ROTATE: every {genome_dict['rebalance']} days hold the top "
                f"{genome_dict['top_n']} ETFs ranked by\n{terms}")
    genome = Genome.from_dict(genome_dict)
    lines = []
    for label, tree in (("ENTRY  (buy when true)", genome.entry),
                        ("EXIT   (sell when true)", genome.exit)):
        lines.append(label)
        lines += _draw(tree, "", True)
    return "\n".join(lines)


def _draw(node: Node, prefix: str, last: bool) -> list[str]:
    branch = "`-- " if last else "|-- "
    if isinstance(node, Compare):
        return [f"{prefix}{branch}{node}"]
    name = type(node).__name__.upper()
    lines = [f"{prefix}{branch}{name}"]
    kids = node.children()
    for i, kid in enumerate(kids):
        lines += _draw(kid, prefix + ("    " if last else "|   "), i == len(kids) - 1)
    return lines
