from evotrader.strategies.base import Strategy
from evotrader.strategies.classic import BuyAndHold, Momentum, RsiMeanReversion, SmaCrossover

REGISTRY: dict[str, type[Strategy]] = {
    cls.name: cls for cls in (BuyAndHold, SmaCrossover, RsiMeanReversion, Momentum)
}

__all__ = ["REGISTRY", "BuyAndHold", "Momentum", "RsiMeanReversion", "SmaCrossover", "Strategy"]
