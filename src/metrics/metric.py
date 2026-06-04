from typing import Any, Dict, Optional


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

    def explainComparison(
        self,
        current_score: Optional[float],
        other_score: Optional[float],
        current_extras: Optional[Dict[str, Any]] = None,
        other_extras: Optional[Dict[str, Any]] = None,
    ) -> str:
        """One-sentence layman-instructor explanation of the score comparison.

        Subclasses should override to inject domain-specific language. The
        generic fallback only compares the two numeric scores.
        """
        if not isinstance(current_score, (int, float)) or not isinstance(other_score, (int, float)):
            return "Comparison not available for this metric."
        delta = float(current_score) - float(other_score)
        if delta > 0.05:
            verb = "above"
        elif delta < -0.05:
            verb = "below"
        else:
            verb = "in line with"
        return f"Your score of {current_score:.2f} is {verb} the reference's {other_score:.2f}."