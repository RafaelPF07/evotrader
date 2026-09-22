"""Strategy genomes: an entry rule and an exit rule, each a boolean expression tree.

    ENTRY (rsi(2) < 15.0 AND trend(200) > 0.010) | EXIT rsi(2) > 70.0

Leaves compare a normalised feature with a threshold; branches combine them with
AND / OR / NOT. Trees are immutable in practice: mutation and crossover always
build new trees, so a genome in the hall of fame can never change under us.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from evotrader.features import KINDS, FeatureSpec, FeatureStore

Path = tuple[int, ...]


class Node:
    def evaluate(self, store: FeatureStore) -> np.ndarray:
        raise NotImplementedError

    def children(self) -> list[Node]:
        return []

    def with_children(self, children: list[Node]) -> Node:
        return self

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @property
    def depth(self) -> int:
        return 1 + max((c.depth for c in self.children()), default=0)

    @property
    def size(self) -> int:
        return 1 + sum(c.size for c in self.children())

    def leaves(self) -> list[Compare]:
        if isinstance(self, Compare):
            return [self]
        return [leaf for c in self.children() for leaf in c.leaves()]


@dataclass(frozen=True)
class Compare(Node):
    feature: FeatureSpec
    op: str  # "<" or ">"
    threshold: float

    def evaluate(self, store: FeatureStore) -> np.ndarray:
        values = store.get(self.feature)
        with np.errstate(invalid="ignore"):  # NaN comparisons are simply False
            return values < self.threshold if self.op == "<" else values > self.threshold

    def to_dict(self) -> dict[str, Any]:
        return {"type": "cmp", "feature": self.feature.kind, "windows": list(self.feature.windows),
                "op": self.op, "threshold": self.threshold}

    def __str__(self) -> str:
        decimals = KINDS[self.feature.kind].decimals
        return f"{self.feature} {self.op} {self.threshold:.{decimals}f}"


@dataclass(frozen=True)
class And(Node):
    left: Node
    right: Node

    def evaluate(self, store: FeatureStore) -> np.ndarray:
        return self.left.evaluate(store) & self.right.evaluate(store)

    def children(self) -> list[Node]:
        return [self.left, self.right]

    def with_children(self, children: list[Node]) -> Node:
        return And(*children)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "and", "left": self.left.to_dict(), "right": self.right.to_dict()}

    def __str__(self) -> str:
        return f"({self.left} AND {self.right})"


@dataclass(frozen=True)
class Or(Node):
    left: Node
    right: Node

    def evaluate(self, store: FeatureStore) -> np.ndarray:
        return self.left.evaluate(store) | self.right.evaluate(store)

    def children(self) -> list[Node]:
        return [self.left, self.right]

    def with_children(self, children: list[Node]) -> Node:
        return Or(*children)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "or", "left": self.left.to_dict(), "right": self.right.to_dict()}

    def __str__(self) -> str:
        return f"({self.left} OR {self.right})"


@dataclass(frozen=True)
class Not(Node):
    child: Node

    def evaluate(self, store: FeatureStore) -> np.ndarray:
        return ~self.child.evaluate(store)

    def children(self) -> list[Node]:
        return [self.child]

    def with_children(self, children: list[Node]) -> Node:
        return Not(children[0])

    def to_dict(self) -> dict[str, Any]:
        return {"type": "not", "child": self.child.to_dict()}

    def __str__(self) -> str:
        return f"NOT {self.child}"


def node_from_dict(d: dict[str, Any]) -> Node:
    match d["type"]:
        case "cmp":
            spec = FeatureSpec(d["feature"], tuple(d["windows"]))
            return Compare(spec, d["op"], float(d["threshold"]))
        case "and":
            return And(node_from_dict(d["left"]), node_from_dict(d["right"]))
        case "or":
            return Or(node_from_dict(d["left"]), node_from_dict(d["right"]))
        case "not":
            return Not(node_from_dict(d["child"]))
    raise ValueError(f"Unknown node type {d['type']!r}")


def subtrees(node: Node, path: Path = ()) -> list[tuple[Path, Node]]:
    out = [(path, node)]
    for i, child in enumerate(node.children()):
        out.extend(subtrees(child, (*path, i)))
    return out


def replace_at(node: Node, path: Path, new: Node) -> Node:
    if not path:
        return new
    kids = node.children()
    kids[path[0]] = replace_at(kids[path[0]], path[1:], new)
    return node.with_children(kids)


@dataclass(frozen=True)
class Genome:
    entry: Node
    exit: Node

    @property
    def size(self) -> int:
        return self.entry.size + self.exit.size

    @property
    def depth(self) -> int:
        return max(self.entry.depth, self.exit.depth)

    def features(self) -> set[FeatureSpec]:
        return {leaf.feature for leaf in self.entry.leaves() + self.exit.leaves()}

    def target_position(self, store: FeatureStore) -> np.ndarray:
        """1.0 from an entry signal until an exit signal, else 0.0 (exit wins ties).

        No entries are allowed until every feature the genome uses has enough
        history, so NOT(<missing data>) can't trigger a trade during warm-up.
        """
        ready = np.ones(len(store), dtype=bool)
        for spec in self.features():
            ready &= ~np.isnan(store.get(spec))
        entry = self.entry.evaluate(store) & ready
        exit_ = self.exit.evaluate(store)

        events = np.full(len(store), np.nan)
        events[entry] = 1.0
        events[exit_] = 0.0
        return pd.Series(events).ffill().fillna(0.0).to_numpy()

    def explain(self, store: FeatureStore, i: int) -> str:
        """Why the rule says what it says on bar `i`: every condition with its live value."""
        parts = []
        for side, tree in (("entry", self.entry), ("exit", self.exit)):
            conds = []
            for leaf in tree.leaves():
                value = store.get(leaf.feature)[i]
                mark = "yes" if bool(leaf.evaluate(store)[i]) else "no"
                conds.append(f"{leaf.feature}={value:.3f} {leaf.op} {leaf.threshold} [{mark}]")
            fired = "FIRED" if bool(tree.evaluate(store)[i]) else "quiet"
            parts.append(f"{side} {fired}: " + "; ".join(conds))
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {"entry": self.entry.to_dict(), "exit": self.exit.to_dict()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Genome:
        return cls(node_from_dict(d["entry"]), node_from_dict(d["exit"]))

    def __str__(self) -> str:
        return f"ENTRY {self.entry} | EXIT {self.exit}"


class GenomeFactory:
    """Random generation and variation operators, driven by one seeded RNG."""

    def __init__(
        self, rng: np.random.Generator, kinds: tuple[str, ...], max_depth: int = 3
    ) -> None:
        self.rng = rng
        self.kinds = kinds
        self.max_depth = max_depth

    # --- generation -------------------------------------------------------
    def random_feature(self) -> FeatureSpec:
        kind = KINDS[str(self.rng.choice(self.kinds))]
        if kind.n_windows == 0:
            return FeatureSpec(kind.name)
        windows = sorted(self.rng.choice(kind.windows, size=kind.n_windows, replace=False))
        return FeatureSpec(kind.name, tuple(int(w) for w in windows))

    def _round(self, kind: str, value: float) -> float:
        k = KINDS[kind]
        return round(float(np.clip(value, k.lo, k.hi)), k.decimals)

    def random_leaf(self) -> Compare:
        spec = self.random_feature()
        k = KINDS[spec.kind]
        threshold = self._round(spec.kind, self.rng.uniform(k.lo, k.hi))
        return Compare(spec, str(self.rng.choice(["<", ">"])), threshold)

    def random_tree(self, max_depth: int) -> Node:
        if max_depth <= 1 or self.rng.random() < 0.4:
            return self.random_leaf()
        roll = self.rng.random()
        if roll < 0.1:
            return Not(self.random_tree(max_depth - 1))
        op = And if roll < 0.7 else Or
        return op(self.random_tree(max_depth - 1), self.random_tree(max_depth - 1))

    def random_genome(self) -> Genome:
        entry = self.random_tree(self.max_depth)
        # A third of the time, exit is simply the opposite of entry ("hold while true").
        exit_ = Not(entry) if self.rng.random() < 0.3 else self.random_tree(self.max_depth)
        return self._fit_depth(Genome(entry, exit_))

    # --- variation --------------------------------------------------------
    def mutate(self, genome: Genome) -> Genome:
        side = "entry" if self.rng.random() < 0.5 else "exit"
        tree: Node = getattr(genome, side)
        path, target = subtrees(tree)[self.rng.integers(len(subtrees(tree)))]
        child = replace(genome, **{side: replace_at(tree, path, self._mutate_node(target))})
        return self._fit_depth(child, fallback=genome)

    def _mutate_node(self, node: Node) -> Node:
        roll = self.rng.random()
        if isinstance(node, Compare):
            k = KINDS[node.feature.kind]
            if roll < 0.45:  # nudge threshold
                step = self.rng.normal(0, 0.1 * (k.hi - k.lo))
                return Compare(node.feature, node.op, self._round(k.name, node.threshold + step))
            if roll < 0.65 and k.n_windows:  # new window lengths, same feature
                return Compare(self._reroll_windows(node.feature), node.op, node.threshold)
            if roll < 0.75:  # flip direction
                return Compare(node.feature, ">" if node.op == "<" else "<", node.threshold)
            if roll < 0.9:  # grow: combine with a new condition
                op = And if self.rng.random() < 0.6 else Or
                return op(node, self.random_leaf())
            return self.random_leaf()
        if roll < 0.4 and node.children():  # shrink: keep one branch
            kids = node.children()
            return kids[self.rng.integers(len(kids))]
        return self.random_tree(2)

    def _reroll_windows(self, spec: FeatureSpec) -> FeatureSpec:
        k = KINDS[spec.kind]
        windows = sorted(self.rng.choice(k.windows, size=k.n_windows, replace=False))
        return FeatureSpec(spec.kind, tuple(int(w) for w in windows))

    def crossover(self, a: Genome, b: Genome) -> Genome:
        """Graft a random subtree of `b` into a random spot in `a` (same side)."""
        side = "entry" if self.rng.random() < 0.5 else "exit"
        tree_a: Node = getattr(a, side)
        spots = subtrees(tree_a)
        donors = subtrees(getattr(b, side))
        path, _ = spots[self.rng.integers(len(spots))]
        _, graft = donors[self.rng.integers(len(donors))]
        child = replace(a, **{side: replace_at(tree_a, path, graft)})
        return self._fit_depth(child, fallback=a)

    def _fit_depth(self, genome: Genome, fallback: Genome | None = None) -> Genome:
        if genome.depth <= self.max_depth + 1:  # +1 leaves room for exit = NOT(entry)
            return genome
        return fallback if fallback is not None else self.random_genome()
