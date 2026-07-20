import itertools
import math

from scoring import ScoringModel


def meeting_block_size(data, name_a, name_b):
    """Smallest power-of-2 block containing both players' lines (same gender only)."""
    la = data.players[name_a]["line"]
    lb = data.players[name_b]["line"]
    gender = data.players[name_a]["gender"]
    size = 2 ** data.gender_max_rounds.get(gender, 7)
    mbs = 2
    while mbs <= size:
        if (la - 1) // mbs == (lb - 1) // mbs:
            return mbs
        mbs *= 2
    return size


def pairwise_cov(data, name_a, name_b, evs, model: ScoringModel):
    """Cov[s_A, s_B] for two players. Zero for cross-gender (independent draws)."""
    if data.players[name_a]["gender"] != data.players[name_b]["gender"]:
        return 0.0
    mbs = meeting_block_size(data, name_a, name_b)
    gender = data.players[name_a]["gender"]
    max_rounds = data.gender_max_rounds.get(gender, 7)
    min_idx = model.prob_start_idx(max_rounds)
    meeting_idx = int(math.log2(mbs)) - 1
    first = max(min_idx, meeting_idx)
    ea = evs[name_a]["all_probs"]
    eb = evs[name_b]["all_probs"]
    cov_per_pt = sum(-ea[i] * eb[i] for i in range(first, max_rounds))
    return model.points_per_round ** 2 * cov_per_pt


def score_variance(data, name, evs, model: ScoringModel):
    """Var[s] for a single player's fantasy score."""
    p = evs[name]
    gender = data.players[name]["gender"]
    max_rounds = data.gender_max_rounds.get(gender, 7)
    min_idx = model.prob_start_idx(max_rounds)
    scoring_probs = p["all_probs"][min_idx:]
    pts = model.points_per_round
    e_s2 = sum((pts ** 2 * (2 * (j + 1) - 1)) * scoring_probs[j] for j in range(len(scoring_probs)))
    return e_s2 - p["ev"] ** 2


def lineup_variance(data, lineup, evs, model: ScoringModel):
    """Portfolio variance of the total lineup score."""
    var = sum(score_variance(data, p, evs, model) for p in lineup)
    for a, b in itertools.combinations(lineup, 2):
        var += 2 * pairwise_cov(data, a, b, evs, model)
    return var


def max_lineup_score(data, lineup, model: ScoringModel):
    """Max points this lineup can score, accounting for players knocking each other out.

    For each round r, at most one lineup member per size-2^r bracket section can
    achieve r wins. So max achievers at round r = # sections of size 2^r with >=1
    lineup member. Sum over all scoring rounds."""
    total = 0
    for gender in ("M", "F"):
        gender_members = [n for n in lineup if data.players[n]["gender"] == gender]
        if not gender_members:
            continue
        max_rounds = data.gender_max_rounds.get(gender, 7)
        min_r = model.min_round(max_rounds)
        lines = [data.players[n]["line"] for n in gender_members]
        for r in range(1, max_rounds + 1):
            if r < min_r:
                continue
            section_size = 2 ** r
            sections_hit = {(ln - 1) // section_size for ln in lines}
            total += model.points_per_round * len(sections_hit)
    return total


def find_top_lineups(data, player_evs, n, excluded, included, token_cap, lineup_size, ev_floor,
                     objective=None):
    """Return (top_lineups, evaluated, ev_histogram) using branch-and-bound DFS.

    top_lineups: list of (score, tuple_of_names), best first
    evaluated: list of (score, tuple_of_names), sorted ascending (full search space)
    ev_histogram: {buckets, percentiles, min, max, n} — orchestrator prints this

    objective: Callable[[lineup, player_evs], float] — ranks lineups. Defaults to gross EV sum.
    The branch-and-bound upper bound uses individual gross EVs, so objective must be
    monotone in individual player EVs (adding a higher-EV player never decreases the score).
    """
    if objective is None:
        objective = lambda lineup, evs: sum(evs[p]["ev"] for p in lineup)

    candidates = sorted(
        [p for p in data.players if data.players[p]["is_priced"] and p not in excluded],
        key=lambda x: player_evs[x]["ev"], reverse=True,
    )
    forced = [p for p in included if p in {c for c in candidates}]
    forced_cost = sum(data.players[p]["cost"] for p in forced)
    forced_ev   = sum(player_evs[p]["ev"]    for p in forced)
    forced_slots = len(forced)
    candidates = [p for p in candidates if p not in included]
    nc = len(candidates)
    costs_arr = [data.players[p]["cost"] for p in candidates]
    evs_arr   = [player_evs[p]["ev"]    for p in candidates]

    # ev_suffix[i] = sum(evs_arr[i:]) — used for EV upper-bound pruning
    ev_suffix = [0.0] * (nc + 1)
    for i in range(nc - 1, -1, -1):
        ev_suffix[i] = ev_suffix[i + 1] + evs_arr[i]

    if lineup_size is not None:
        min_size = max(0, lineup_size - forced_slots)
        max_size = max(0, lineup_size - forced_slots)
    else:
        min_size = 1
        max_size = nc + forced_slots

    def run_search(prune_floor, collect_all=False):
        results = []
        evaluated = []

        def nth_best():
            return results[-1][0] if len(results) == n else -1.0

        def cutoff():
            return prune_floor if collect_all else max(nth_best(), prune_floor)

        def search(idx, combo, cost, ev):
            size = len(combo)

            if size >= min_size:
                full_combo = tuple(forced) + tuple(combo)
                score = objective(full_combo, player_evs)
                evaluated.append((score, full_combo))
                cur_nth = nth_best()
                if score > cur_nth:
                    if len(results) < n:
                        results.append((score, full_combo))
                        if len(results) == n:
                            results.sort(key=lambda x: -x[0])
                    else:
                        results[-1] = (score, full_combo)
                        results.sort(key=lambda x: -x[0])

            if size == max_size:
                return

            remaining = nc - idx
            min_more  = max(0, min_size - size)
            max_slots = max_size - size

            if remaining < min_more:
                return

            ev_ub = ev + ev_suffix[idx] - ev_suffix[min(idx + max_slots, nc)]
            if forced_ev + ev_ub <= cutoff():
                return

            for i in range(idx, nc):
                if nc - i - 1 < max(0, min_size - size - 1):
                    break

                new_cost = cost + costs_arr[i]
                if lineup_size is None and new_cost > token_cap:
                    continue

                slots_after = max_slots - 1
                ev_ub_i = ev + evs_arr[i] + ev_suffix[i + 1] - ev_suffix[min(i + 1 + slots_after, nc)]
                if forced_ev + ev_ub_i <= cutoff():
                    continue

                combo.append(candidates[i])
                search(i + 1, combo, new_cost, ev + evs_arr[i])
                combo.pop()

        search(0, [], forced_cost, 0.0)
        results.sort(key=lambda x: -x[0])
        return results, evaluated

    # Pass 1: find optimal EV
    results, evaluated = run_search(-1.0)

    # Pass 2: if ev_floor set, re-run keeping all lineups within floor of optimal
    if ev_floor > 0 and results:
        optimal_ev = results[0][0]
        floor = optimal_ev - ev_floor
        _, evaluated = run_search(floor, collect_all=True)
        results = [(ev, combo) for ev, combo in evaluated if ev >= floor]
        results.sort(key=lambda x: -x[0])
    results.sort(key=lambda x: -x[0])
    evaluated.sort(key=lambda x: x[0])

    evs_only = [e for e, _ in evaluated]
    ne = len(evs_only)
    buckets = {}
    for ev in evs_only:
        b = round(ev * 2) / 2
        buckets[b] = buckets.get(b, 0) + 1
    percentile_marks = [10, 25, 50, 75, 90]
    pct_vals = {p: evs_only[min(int(p / 100 * ne), ne - 1)] for p in percentile_marks}

    ev_histogram = {
        "buckets": buckets,
        "percentiles": pct_vals,
        "min": evs_only[0] if evs_only else 0.0,
        "max": evs_only[-1] if evs_only else 0.0,
        "n": ne,
    }

    return results[:n], evaluated, ev_histogram
