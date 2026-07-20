import math

from formatting import pct, fmt_player
from optimizer import lineup_variance, score_variance, max_lineup_score
from scoring import ScoringModel
from bracket import (
    bracket_opponent_lines,
    section_win_probs,
    round_label as _round_label,
)
from elo_math import calculate_match_win_prob


def pool_section_rows(data, player_evs, gender, elo_col, top_evtok_names=None):
    """Row lists for the player pool table of one gender. Name is plain (no markup)."""
    if top_evtok_names is None:
        top_evtok_names = set()
    names = sorted(
        [n for n, p in data.players.items() if p["is_priced"] and p["gender"] == gender],
        key=lambda n: player_evs[n]["ev"],
        reverse=True,
    )
    rows = []
    for name in names:
        pd = data.players[name]
        s = player_evs[name]
        f = fmt_player(pd, s, elo_col)
        evtok = f["ev_tok"] + "*" if name in top_evtok_names else f["ev_tok"]
        rows.append([name, pd["cost"], f["elo"],
                     f["p_qf"], f["p_sf"], f["p_f"], f["p_ch"],
                     f["ev"], evtok, f["draw_eff"]])
    return rows


def lineup_summary(data, player_evs, lineup, model, token_cap, elo_col):
    """Summary dict for a lineup card header."""
    gross_ev = round(sum(player_evs[p]["ev"] for p in lineup), 3)
    tokens = sum(data.players[p]["cost"] for p in lineup)
    portfolio_std = math.sqrt(max(lineup_variance(data, lineup, player_evs, model), 0))
    winner_probs = {}
    for p in lineup:
        g = data.players[p]["gender"]
        winner_probs[g] = winner_probs.get(g, 0.0) + player_evs[p]["p_ch"]
    p_any_winner = 1.0 - math.prod(1.0 - v for v in winner_probs.values())
    return {
        "gross_ev": gross_ev,
        "portfolio_std": portfolio_std,
        "tokens": tokens,
        "token_cap": token_cap,
        "p_any_winner": p_any_winner,
        "max_score": max_lineup_score(data, lineup, model),
        "elo_label": {"elo": "Elo", "gelo": "gElo", "celo": "cElo", "helo": "hElo"}.get(elo_col, elo_col),
    }


def lineup_rows(data, player_evs, lineup, model, elo_col):
    """Row lists for a lineup card table.
    Quadrant is stored as int; output layer formats it as Q{n} or Quarter {n}."""
    rows = []
    for p in lineup:
        pd = data.players[p]
        s = player_evs[p]
        f = fmt_player(pd, s, elo_col)
        indiv_std = math.sqrt(max(score_variance(data, p, player_evs, model), 0))
        rows.append([p, pd["gender"], pd["cost"], f["elo"],
                     f["p_qf"], f["p_sf"], f["p_f"], f["p_ch"],
                     f["ev"], f["ev_tok"], f"{indiv_std:.2f}", pd["quadrant"], f["draw_eff"]])
    return rows


def draw_efficiency_rows(data, player_evs, elo_col):
    """Row lists for the draw efficiency table."""
    priced = [
        n for n, pd in data.players.items()
        if pd["is_priced"] and player_evs[n].get("draw_eff") is not None
    ]
    priced.sort(key=lambda n: player_evs[n]["draw_eff"], reverse=True)
    rows = []
    for name in priced:
        pd = data.players[name]
        s = player_evs[name]
        f = fmt_player(pd, s, elo_col)
        neutral_ev = s.get("neutral_ev")
        n_ev_str = f"{neutral_ev:.2f}" if neutral_ev is not None else "—"
        n_evtok_str = f"{neutral_ev/pd['cost']:.2f}" if neutral_ev is not None else "—"
        rows.append([name, pd["gender"], pd["cost"], f["elo"],
                     n_ev_str, f["ev"], n_evtok_str, f["ev_tok"], f["draw_eff"]])
    return rows


def sim_comparison_rows(data, lineups, sim_scores, player_evs, model, labels=None):
    """Row lists for the analytical vs simulated comparison table."""
    rows = []
    for i, (ev, lineup) in enumerate(lineups):
        label = labels[i] if labels else f"#{i+1}"
        exp_std = math.sqrt(max(lineup_variance(data, lineup, player_evs, model), 0))
        scores = sim_scores[i]
        n = len(scores)
        sim_mean = sum(scores) / n
        sim_std = math.sqrt(sum((s - sim_mean) ** 2 for s in scores) / n)
        rows.append([label,
                     f"{ev:.2f}", f"{exp_std:.2f}",
                     f"{sim_mean:.2f}", f"{sim_std:.2f}",
                     f"{sim_mean - ev:+.2f}"])
    return rows


def score_distribution_rows(sim_scores, labels=None):
    """Build (p_rows, ge_rows, headers) for score distribution tables."""
    n = len(sim_scores[0])
    max_score = max(max(s) for s in sim_scores)
    scores_range = range(0, max_score + 2, 2)
    lineup_labels = labels if labels else [f"#{i+1}" for i in range(len(sim_scores))]

    freq = []
    for scores in sim_scores:
        fd = {}
        for s in scores:
            fd[s] = fd.get(s, 0) + 1
        freq.append(fd)

    p_rows = []
    for k in scores_range:
        vals = [fd.get(k, 0) / n * 100 for fd in freq]
        if max(vals) < 0.1:
            continue
        p_rows.append([str(k)] + [f"{v:.1f}%" for v in vals])

    ge_rows = []
    prev_vals = None
    for k in scores_range:
        if k == 0:
            continue
        vals = [sum(fd.get(s, 0) for s in scores_range if s >= k) / n * 100 for fd in freq]
        if all(v == 0.0 for v in vals):
            continue
        if prev_vals is None or any(abs(v - pv) >= 0.1 for v, pv in zip(vals, prev_vals)):
            ge_rows.append([str(k)] + [f"{v:.1f}%" for v in vals])
            prev_vals = vals

    return p_rows, ge_rows, ["Score"] + lineup_labels


def analysis_rows(evaluated, top_k=100):
    """Build (freq_rows, pair_rows, summary_str) for player frequency analysis."""
    if not evaluated:
        return [], [], ""
    members = evaluated[-top_k:]
    nm = len(members)
    evs = [ev for ev, _ in members]
    avg_ev = sum(evs) / nm
    std_ev = math.sqrt(sum((e - avg_ev) ** 2 for e in evs) / nm)
    summary_str = (f"top {nm} lineups  "
                   f"(EV {evs[0]:.2f} – {evs[-1]:.2f},  avg {avg_ev:.2f},  σ {std_ev:.2f})")

    all_players = sorted({p for _, combo in members for p in combo})
    freq = {p: sum(1 for _, combo in members if p in combo) for p in all_players}
    freq_sorted = sorted(freq.items(), key=lambda x: -x[1])
    freq_rows = [[p, str(f), f"{f/nm*100:.0f}%"] for p, f in freq_sorted if f / nm >= 0.20]

    freq_frac = {p: f / nm for p, f in freq_sorted}
    common = [p for p, _ in freq_sorted]
    pair_rows = []
    for i, pa in enumerate(common):
        for pb in common[i+1:]:
            both = sum(1 for _, combo in members if pa in combo and pb in combo)
            if both >= 2:
                expected = freq_frac[pa] * freq_frac[pb]
                lift = both / nm / expected if expected > 0 else 0.0
                pair_rows.append([f"{pa} + {pb}", str(both), f"{lift:.2f}"])
    pair_rows.sort(key=lambda r: -float(r[2]))
    return freq_rows, pair_rows[:20], summary_str


def best_player_at_rows(priced, scores, player_evs, data, model):
    """Row lists for the best-single-pick-by-threshold table."""
    max_score = model.max_score(7)
    thresholds = [k for k in [2, 4, 6, 8, 10, 12, 14] if k <= max_score]
    n_trials = len(next(iter(scores.values())))
    rows = []
    for k in thresholds:
        ge = {name: sum(1 for s in scores[name] if s >= k) / n_trials * 100 for name in priced}
        best = max(priced, key=lambda n: ge[n])
        rows.append([f"≥{k}", f"{ge[best]:.1f}%", f"{player_evs[best]['ev']:.2f}",
                     str(data.players[best]["cost"]), best])
    return rows


def path_rows(data, player_name, player_ev, advancements, bo5, model, elo_col):
    """Build (rows, headers, header_str) for the path analysis table."""
    p_data = data.players[player_name]
    gender = p_data["gender"]
    p_elo = p_data["elo"]
    line = p_data["line"]
    quad = p_data["quadrant"]
    max_rounds = data.gender_max_rounds.get(gender, 7)
    size = 2 ** max_rounds
    elo_label = {"elo": "Elo", "gelo": "gElo", "celo": "cElo", "helo": "hElo"}.get(elo_col, elo_col)

    cost_str = f"Cost {p_data['cost']}  |  " if p_data["is_priced"] else ""
    ev_val = player_ev["ev"]
    ev_tok_str = f"  |  EV/Tok {ev_val/p_data['cost']:.2f}" if p_data["is_priced"] else ""
    header_str = (f"Path: {player_name}  (line {line}, Q{quad}, {gender})  "
                  f"{elo_label} {round(p_elo)}  |  {cost_str}EV {ev_val:.2f}{ev_tok_str}")

    fb_starts = {1: 1500.0, 2: 1520.0, 3: 1510.0, 4: 1530.0}
    fb_end = 1950.0
    start = fb_starts[quad]
    fb = ([fb_end] if max_rounds == 1
          else [start + (fb_end - start) * i / (max_rounds - 1) for i in range(max_rounds)])

    opp_sections = bracket_opponent_lines(line, size)
    scoring_start = model.min_round(max_rounds)
    round_defs = [
        (rnd, _round_label(rnd, max_rounds), opp_sections[rnd - 1], fb[rnd - 1],
         rnd > scoring_start)
        for rnd in range(1, max_rounds + 1)
    ]
    round_defs.append((max_rounds + 1, "W", [], None, (max_rounds + 1) > scoring_start))

    advance_through = advancements.get(player_name)
    p_reach = 1.0
    result_rows = []

    for rnd, rnd_name, opp_lines, fallback, scores_at in round_defs:
        if rnd_name == "W":
            win_p, opp_str = 1.0, "—"
        elif advance_through is not None and rnd <= advance_through:
            win_p = 1.0
            probs = section_win_probs(data, opp_lines, gender, advancements, bo5)
            real_probs = {j: p_j for j, p_j in probs.items()
                         if data.line_index[(gender, j)]["elo"] > 0.0}
            if not real_probs:
                continue
            top3 = sorted(real_probs.items(), key=lambda x: -x[1])[:3]
            opp_str = " / ".join(
                f"{data.line_to_name.get((gender, j), f'line {j}')} "
                f"({round(data.line_index[(gender, j)]['elo'])}, {pct(p_j)}%)"
                for j, p_j in top3
            ) + "  [forced]"
        else:
            probs = section_win_probs(data, opp_lines, gender, advancements, bo5)
            if probs:
                real_probs = {j: p_j for j, p_j in probs.items()
                             if data.line_index[(gender, j)]["elo"] > 0.0}
                if not real_probs:
                    p_reach *= 1.0
                    continue
                win_p = sum(
                    p_j * (0.0 if (advancements.get(data.line_to_name.get((gender, j))) or -1) >= rnd
                           else calculate_match_win_prob(p_elo, data.line_index[(gender, j)]["elo"],
                                                         bo5=bo5, gender=gender))
                    for j, p_j in probs.items()
                )
                top3 = sorted(real_probs.items(), key=lambda x: -x[1])[:3]
                opp_str = " / ".join(
                    f"{data.line_to_name.get((gender, j), f'line {j}')} "
                    f"({round(data.line_index[(gender, j)]['elo'])}, {pct(p_j)}%)"
                    for j, p_j in top3
                )
            else:
                win_p = calculate_match_win_prob(p_elo, fallback, bo5=bo5, gender=gender)
                opp_str = f"unknown (fallback {elo_label} {round(fallback)})"

        p_reach_next = p_reach * win_p
        rnd_pts = model.points_per_round * p_reach if scores_at else 0.0
        result_rows.append([
            rnd_name, opp_str,
            pct(win_p) + "%",
            pct(p_reach) + "%",
            f"{rnd_pts:.2f}" if rnd_pts > 0 else "—",
        ])
        p_reach = p_reach_next

    headers = ["Round", "Opponent(s)  (Elo, P(faces you))", "Win%", "P(reach)", "E[pts]"]
    return result_rows, headers, header_str


def path_sim_rows(player_name, rounds_reached, score_counts, all_probs, n_trials,
                  data, opp_sections, advancements, bo5):
    """Build (rows1, p_rows, ge_rows) for path simulation tables."""
    p_data = data.players[player_name]
    gender = p_data["gender"]
    max_rounds = data.gender_max_rounds.get(gender, 7)

    def is_bye(opp_lines):
        probs = section_win_probs(data, opp_lines, gender, advancements, bo5)
        return not any(data.line_index[(gender, j)]["elo"] > 0.0 for j in probs)

    rounds_reached[0] = n_trials
    rows1 = []
    for rnd in range(1, max_rounds + 1):
        if is_bye(opp_sections[rnd - 1]):
            continue
        label = _round_label(rnd, max_rounds)
        sim_pct = rounds_reached[rnd - 1] / n_trials
        ana_pct = all_probs[rnd - 2] if rnd > 1 else 1.0
        rows1.append([label, pct(sim_pct) + "%", pct(ana_pct) + "%",
                      f"{(sim_pct - ana_pct)*100:+.1f}pp"])
    sim_w = rounds_reached[max_rounds] / n_trials
    ana_w = all_probs[max_rounds - 1]
    rows1.append(["W", pct(sim_w) + "%", pct(ana_w) + "%",
                  f"{(sim_w - ana_w)*100:+.1f}pp"])

    max_score = max(score_counts.keys()) if score_counts else 0
    scores_range = range(0, max_score + 2, 2)
    p_rows = [[str(k), f"{score_counts.get(k, 0)/n_trials*100:.1f}%"]
              for k in scores_range if score_counts.get(k, 0) / n_trials * 100 >= 0.1]
    ge_rows = []
    prev = None
    for k in scores_range:
        if k == 0:
            continue
        v = sum(score_counts.get(s, 0) for s in scores_range if s >= k) / n_trials * 100
        if v == 0:
            continue
        if prev is None or abs(v - prev) >= 0.1:
            ge_rows.append([str(k), f"{v:.1f}%"])
            prev = v
    return rows1, p_rows, ge_rows
