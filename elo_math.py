import math


def _generate_elo_bo5_lookup(max_diff=1000):
    lookup = [0] * (max_diff + 1)
    for diff in range(max_diff + 1):
        p_bo3 = 1.0 / (1.0 + 10 ** ((-diff) / 400.0))
        low, high = 0.0, 1.0
        for _ in range(35):
            p = (low + high) / 2.0
            if (3 * p**2 - 2 * p**3) < p_bo3:
                low = p
            else:
                high = p
        p_bo5 = 10 * p**3 - 15 * p**4 + 6 * p**5
        if p_bo5 >= 1.0:
            adjusted_diff = max_diff * 1.5
        elif p_bo5 <= 0.0:
            adjusted_diff = 0
        else:
            adjusted_diff = -400 * math.log10(1.0 / p_bo5 - 1.0)
        lookup[diff] = round(adjusted_diff)
    return lookup


_MAX_ELO_DIFF = 1000
_ELO_BO5_LOOKUP = _generate_elo_bo5_lookup(_MAX_ELO_DIFF)


def calculate_match_win_prob(elo_a, elo_b, *, bo5=False, gender=None):
    if elo_b == 0.0:
        return 1.0
    if elo_a == 0.0:
        return 0.0
    if bo5 and gender == "M":
        raw_diff = elo_a - elo_b
        abs_diff = min(abs(round(raw_diff)), _MAX_ELO_DIFF)
        adjusted_diff = _ELO_BO5_LOOKUP[abs_diff]
        if raw_diff < 0:
            adjusted_diff = -adjusted_diff
        return 1.0 / (1.0 + 10 ** (-adjusted_diff / 400.0))
    return 1 / (1 + math.pow(10, (elo_b - elo_a) / 400))
