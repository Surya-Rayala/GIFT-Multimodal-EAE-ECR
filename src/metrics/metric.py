from typing import Any, Dict


class AbstractMetric:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config or {}
        self.metricName = "Metric"

    def process(self, ctx) -> None:
        raise NotImplementedError("Subclasses must implement process(ctx)")

    def updateFrame(self, map_pos) -> None:
        pass

    def getFinalScore(self) -> float:
        raise NotImplementedError("Subclasses must implement getFinalScore()")

    @staticmethod
    def expertCompare(*args, **kwargs):
        raise NotImplementedError("Subclasses must implement expertCompare(...)")