import math
import random

from elo_math import calculate_match_win_prob
from scoring import ScoringModel


def simulate_tournament(data, gender, rng, advancements, bo5, k_factor=0, live_elos=None):
    """Simulate one full gender draw. Returns {player_name: rounds_won} for all players.
    live_elos, if provided, is a mutable {name: elo} dict updated each match (K-factor mode)."""
    survivors = {
        pd["line"]: name
        for name, pd in data.players.items()
        if pd["gender"] == gender
    }
    rounds_won = {
        name: 0 for name, pd in data.players.items()
        if pd["gender"] == gender and not name.startswith("__BYE_")
    }

    n_players = len(survivors)
    max_rounds = data.gender_max_rounds.get(gender, 7)
    current_round = max_rounds - int(math.log2(n_players)) + 1
    while n_players > 1:
        next_survivors = {}
        lines = sorted(survivors)
        for i in range(0, len(lines), 2):
            la, lb = lines[i], lines[i + 1]
            na, nb = survivors[la], survivors[lb]
            elo_a = live_elos[na] if live_elos else data.players[na]["elo"]
            elo_b = live_elos[nb] if live_elos else data.players[nb]["elo"]
            adv_a = advancements.get(na)
            adv_b = advancements.get(nb)
            force_a = adv_a is not None and current_round <= adv_a
            force_b = adv_b is not None and current_round <= adv_b
            if force_a and not force_b:
                winner_line, winner, loser = la, na, nb
                rounds_won[winner] += 1
                next_survivors[winner_line] = winner
            elif force_b and not force_a:
                winner_line, winner, loser = lb, nb, na
                rounds_won[winner] += 1
                next_survivors[winner_line] = winner
            elif elo_a == 0.0:
                next_survivors[lb] = nb
                if nb in rounds_won:
                    rounds_won[nb] += 1
            elif elo_b == 0.0:
                next_survivors[la] = na
                if na in rounds_won:
                    rounds_won[na] += 1
            else:
                p = calculate_match_win_prob(elo_a, elo_b, bo5=bo5, gender=gender)
                if rng.random() < p:
                    winner_line, winner, loser, p_win = la, na, nb, p
                else:
                    winner_line, winner, loser, p_win = lb, nb, na, 1 - p
                rounds_won[winner] += 1
                next_survivors[winner_line] = winner
                if live_elos is not None:
                    live_elos[winner] += k_factor * (1 - p_win)
                    live_elos[loser]  += k_factor * (0 - p_win)
        survivors = next_survivors
        n_players = len(survivors)
        current_round += 1

    return rounds_won


def score_lineup_from_sim(data, lineup, rounds_won, model: ScoringModel):
    """Score a lineup against one simulated tournament result."""
    score = 0
    for name in lineup:
        r = rounds_won.get(name, 0)
        gender = data.players[name]["gender"]
        max_rounds = data.gender_max_rounds.get(gender, 7)
        score += model.score(r, max_rounds)
    return score


_SIM_CALL_CAP = 10_000_000


def cap_sim_pool(pool, n_trials, label=""):
    cap = max(1, _SIM_CALL_CAP // n_trials)
    if len(pool) > cap:
        print(
            f"WARNING: {label}{len(pool):,} lineups × {n_trials:,} trials = "
            f"{len(pool)*n_trials:,} calls — capping to top {cap:,} by EV "
            f"(>{len(pool)-cap:,} dropped).",
            flush=True,
        )
        pool = pool[-cap:]
    return pool


def run_simulations(data, lineups, n_trials, advancements, bo5, k_factor, model: ScoringModel):
    """Run n_trials full-draw simulations. Returns list of sorted score lists, one per lineup."""
    rng = random.Random()
    scores = [[] for _ in lineups]
    use_live_elos = k_factor != 0
    base_elos = {name: pd["elo"] for name, pd in data.players.items()} if use_live_elos else None

    for _ in range(n_trials):
        if use_live_elos:
            live_elos = dict(base_elos)
            m_result = simulate_tournament(data, "M", rng, advancements, bo5, k_factor, live_elos)
            f_result = simulate_tournament(data, "F", rng, advancements, bo5, k_factor, live_elos)
        else:
            m_result = simulate_tournament(data, "M", rng, advancements, bo5)
            f_result = simulate_tournament(data, "F", rng, advancements, bo5)
        combined = {**m_result, **f_result}
        for i, (_, lineup) in enumerate(lineups):
            scores[i].append(score_lineup_from_sim(data, lineup, combined, model))

    for s in scores:
        s.sort()
    return scores
