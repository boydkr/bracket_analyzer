import math

from elo_math import calculate_match_win_prob
from scoring import ScoringModel


def label_to_round(label, max_rounds):
    """Convert a round label to the last forced-win round index (1-based).
    "QF" means the player is guaranteed to *reach* the QF — forced to win all rounds before it."""
    label = label.upper()
    if label in ("W", "WIN", "WINNER"):
        return max_rounds
    if label == "F":
        return max(1, max_rounds - 1)
    if label == "SF":
        return max(1, max_rounds - 2)
    if label == "QF":
        return max(1, max_rounds - 3)
    if label.startswith("R") and label[1:].isdigit():
        size = int(label[1:])
        return max_rounds - int(math.log2(size))
    raise ValueError(f"Unknown round label '{label}'. Use R128/R64/R32/R16/QF/SF/F/W.")


def round_label(rnd, max_rounds):
    """Label for the rnd-th match played in a max_rounds-round draw.
    rnd=1 is the first match (R128 in a 128-draw), rnd=max_rounds is the Final."""
    rounds_remaining_before = max_rounds - rnd + 1
    if rounds_remaining_before == 1: return "F"
    if rounds_remaining_before == 2: return "SF"
    if rounds_remaining_before == 3: return "QF"
    return f"R{2**rounds_remaining_before}"


def bracket_opponent_lines(line, size):
    """Return tuple of opponent sections for a draw of the given size.
    Section i (0-indexed) is the block of lines that could face `line` in round i+1."""
    def block(ln, s):
        return ((ln - 1) // s) * s + 1

    def sibling(ln, s):
        my_start = block(ln, s)
        parent_start = block(ln, s * 2)
        return (parent_start + s) if my_start == parent_start else parent_start

    sections = []
    s = 1
    while s <= size // 2:
        sib = sibling(line, s)
        sections.append([sib] if s == 1 else list(range(sib, sib + s)))
        s *= 2
    return tuple(sections)


def section_win_probs(data, lines, gender, advancements, bo5, first_round=1):
    """Return {line: P(that player wins the section)} using recursive bracket simulation.
    The section must be a power-of-two-sized contiguous block.
    Reads and writes data.section_cache for memoization."""
    key = (gender, lines[0], lines[-1], first_round)
    if key in data.section_cache:
        return data.section_cache[key]
    known = [l for l in lines if (gender, l) in data.line_index]
    if not known:
        data.section_cache[key] = {}
        return {}
    if len(lines) == 1:
        result = {lines[0]: 1.0} if known else {}
        data.section_cache[key] = result
        return result
    if len(lines) == 2:
        a, b = lines[0], lines[1]
        a_known = (gender, a) in data.line_index
        b_known = (gender, b) in data.line_index
        if a_known and b_known:
            adv_a = advancements.get(data.line_to_name.get((gender, a)))
            adv_b = advancements.get(data.line_to_name.get((gender, b)))
            force_a = adv_a is not None and first_round <= adv_a
            force_b = adv_b is not None and first_round <= adv_b
            if force_a and not force_b:
                result = {a: 1.0, b: 0.0}
            elif force_b and not force_a:
                result = {a: 0.0, b: 1.0}
            else:
                elo_a = data.line_index[(gender, a)]["elo"]
                elo_b = data.line_index[(gender, b)]["elo"]
                p = calculate_match_win_prob(elo_a, elo_b, bo5=bo5, gender=gender)
                result = {a: p, b: 1 - p}
        elif a_known:
            result = {a: 1.0}
        else:
            result = {b: 1.0}
        data.section_cache[key] = result
        return result

    mid = len(lines) // 2
    left, right = lines[:mid], lines[mid:]
    section_rounds = int(math.log2(len(lines)))
    left_probs  = section_win_probs(data, left, gender, advancements, bo5, first_round)
    right_probs = section_win_probs(data, right, gender, advancements, bo5, first_round)
    cross_round = first_round + section_rounds - 1

    result = {}
    for l, p_l in left_probs.items():
        elo_l = data.line_index[(gender, l)]["elo"]
        adv_l = advancements.get(data.line_to_name.get((gender, l)))
        force_l = adv_l is not None and cross_round <= adv_l
        for r, p_r in right_probs.items():
            elo_r = data.line_index[(gender, r)]["elo"]
            adv_r = advancements.get(data.line_to_name.get((gender, r)))
            force_r = adv_r is not None and cross_round <= adv_r
            if force_l and not force_r:
                p_lr = 1.0
            elif force_r and not force_l:
                p_lr = 0.0
            else:
                p_lr = calculate_match_win_prob(elo_l, elo_r, bo5=bo5, gender=gender)
            result[l] = result.get(l, 0) + p_l * p_r * p_lr
            result[r] = result.get(r, 0) + p_l * p_r * (1 - p_lr)
    data.section_cache[key] = result
    return result


def expected_win_prob(data, player_elo, lines, gender, advancements, bo5, fallback_elo,
                      first_round=1, facing_round=None):
    """Σ P(j wins section) × P(player beats j).
    facing_round is the round at which the player actually meets the section winner;
    if an opponent has a forced advance covering that round, win prob against them is 0."""
    if facing_round is None:
        facing_round = first_round
    probs = section_win_probs(data, lines, gender, advancements, bo5, first_round)
    if not probs:
        return calculate_match_win_prob(player_elo, fallback_elo, bo5=bo5, gender=gender)
    total = 0.0
    for j, p_j in probs.items():
        adv_j = advancements.get(data.line_to_name.get((gender, j)))
        if adv_j is not None and facing_round <= adv_j:
            p_win = 0.0
        else:
            p_win = calculate_match_win_prob(player_elo, data.line_index[(gender, j)]["elo"],
                                              bo5=bo5, gender=gender)
        total += p_j * p_win
    return total


def compute_ev(data, player_name, advancements, bo5, model: ScoringModel):
    """Simulate a player's path through the bracket using actual draw opponents
    where available, falling back to generic tier Elos otherwise."""
    p_data = data.players[player_name]
    gender = p_data["gender"]
    p_elo = p_data["elo"]
    line = p_data["line"]
    quad = p_data["quadrant"]
    max_rounds = data.gender_max_rounds.get(gender, 7)
    size = 2 ** max_rounds

    fb_starts = {1: 1500.0, 2: 1520.0, 3: 1510.0, 4: 1530.0}
    fb_end = 1950.0
    start = fb_starts[quad]
    if max_rounds == 1:
        fb = [fb_end]
    else:
        fb = [start + (fb_end - start) * i / (max_rounds - 1) for i in range(max_rounds)]

    opp_sections = bracket_opponent_lines(line, size)
    p_reach = 1.0
    all_p = []
    advance_through = advancements.get(player_name)
    for i, opp_lines in enumerate(opp_sections):
        rnd = i + 1
        if advance_through is not None and rnd <= advance_through:
            win_p = 1.0
        else:
            win_p = expected_win_prob(data, p_elo, opp_lines, gender, advancements, bo5,
                                      fb[i], first_round=1, facing_round=rnd)
        p_reach = p_reach * win_p
        all_p.append(round(p_reach, 4))

    min_idx = model.prob_start_idx(max_rounds)
    ev = round(model.points_per_round * sum(all_p[min_idx:]), 3)
    return {
        "p_qf": all_p[-4] if len(all_p) >= 4 else all_p[0],
        "p_sf": all_p[-3] if len(all_p) >= 3 else all_p[0],
        "p_f":  all_p[-2] if len(all_p) >= 2 else all_p[0],
        "p_ch": all_p[-1],
        "ev":        ev,
        "all_probs": all_p,
    }


def compute_draw_efficiency(data, player_name, player_evs, model: ScoringModel, bo5):
    """EV / neutral_EV. > 1.0 = favorable draw, < 1.0 = tough draw."""
    p_data = data.players[player_name]
    gender = p_data["gender"]
    p_elo = p_data["elo"]
    max_rounds = data.gender_max_rounds.get(gender, 7)

    real_others = [
        n for n, pd in data.players.items()
        if pd["gender"] == gender
        and not n.startswith("__BYE_")
        and n != player_name
        and pd["elo"] > 0
    ]

    p_reach = 1.0
    neutral_all_p = []
    for r in range(1, max_rounds + 1):
        weights, elos = [], []
        for j in real_others:
            if j not in player_evs:
                continue
            j_probs = player_evs[j]["all_probs"]
            p_j_here = 1.0 if r == 1 else j_probs[r - 2]
            if p_j_here > 0:
                weights.append(p_j_here)
                elos.append(data.players[j]["elo"])
        if weights:
            total_w = sum(weights)
            neutral_opp_elo = sum(w * e for w, e in zip(weights, elos)) / total_w
            win_p = calculate_match_win_prob(p_elo, neutral_opp_elo, bo5=bo5, gender=gender)
        else:
            win_p = 0.5
        p_reach *= win_p
        neutral_all_p.append(p_reach)

    min_idx = model.prob_start_idx(max_rounds)
    neutral_ev = model.points_per_round * sum(neutral_all_p[min_idx:])
    actual_ev = player_evs[player_name]["ev"]
    if neutral_ev == 0:
        return None, None
    return actual_ev / neutral_ev, neutral_ev
