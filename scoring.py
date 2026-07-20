from dataclasses import dataclass


@dataclass
class ScoringModel:
    """Contest point scheme. Default: 2 pts per threshold for final N rounds + win."""
    final_rounds: int
    points_per_round: int = 2

    def min_round(self, max_rounds: int) -> int:
        """First round (1-based) that awards points when won."""
        return max_rounds - self.final_rounds

    def prob_start_idx(self, max_rounds: int) -> int:
        """0-based index into an all_probs list where scoring begins.
        all_probs[i] = P(winning i+1 matches), so scoring starts at index min_round-1."""
        return max(0, self.min_round(max_rounds) - 1)

    def scoring_thresholds(self, max_rounds: int) -> range:
        """Rounds (1-based) where winning awards points."""
        return range(self.min_round(max_rounds), max_rounds + 1)

    def score(self, rounds_won: int, max_rounds: int) -> int:
        """Total points for a player who won rounds_won matches."""
        return sum(
            self.points_per_round
            for t in self.scoring_thresholds(max_rounds)
            if rounds_won >= t
        )

    def max_score(self, max_rounds: int) -> int:
        return self.points_per_round * (self.final_rounds + 1)

    @staticmethod
    def from_final_rounds(n: int, points_per_round: int = 2) -> "ScoringModel":
        return ScoringModel(final_rounds=n, points_per_round=points_per_round)
