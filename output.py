from dataclasses import dataclass

from formatting import fixed_table, pct


@dataclass
class OutputConfig:
    discord: bool


def _discord_table(title, rows, headers, cfg, note=None, preamble=None):
    if title:
        print(f"**{title}**")
    if note:
        print(f"_{note}_")
    if preamble:
        print(preamble)
    print("```")
    print("\n".join(fixed_table(headers, rows)))
    print("```")


def _md_table(title, rows, headers, cfg, note=None, preamble=None, header_level="##"):
    if title:
        print(f"{header_level} {title}\n")
    if note:
        print(f"_{note}_\n")
    if preamble:
        print(preamble + "\n")
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(str(c) for c in row) + " |")
    print()


def print_pool_section(rows, gender_title, elo_label, cfg):
    """Print one gender's player pool table."""
    headers = ["Player", "Cost", elo_label, "QF%", "SF%", "F%", "W%", "EV", "EV/Tok", "DrawEff"]
    if cfg.discord:
        if gender_title:
            print(f"**{gender_title}**")
        print("```")
        print("\n".join(fixed_table(headers, rows)))
        print("```")
    else:
        if gender_title:
            print(f"### {gender_title}\n")
        print(f"| Player | Cost | {elo_label} | QF% | SF% | F% | W% | EV | EV/Token | DrawEff |")
        print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for row in rows:
            name, cost, elo, p_qf, p_sf, p_f, p_ch, ev, evtok, draw_eff = row
            print(f"| **{name}** | {cost} | {elo} "
                  f"| {p_qf} | {p_sf} | {p_f} | {p_ch} "
                  f"| {ev} | {evtok} | {draw_eff} |")
        print()


def print_lineup(rows, summary, title, note, cfg):
    """Print a lineup card with header summary line."""
    elo_label = summary["elo_label"]
    winner_str = (f"P(winner): {pct(summary['p_any_winner'])}%  |  "
                  f"Max: {summary['max_score']}")
    headers = ["Player", "G", "Cost", elo_label,
               "QF%", "SF%", "F%", "W%", "EV", "EV/Tok", "StdDev", "Quad", "DrawEff"]

    if cfg.discord:
        display_rows = []
        for row in rows:
            name, g, cost, elo, p_qf, p_sf, p_f, p_ch, ev, evtok, std, quad, draw_eff = row
            display_rows.append([name, g, cost, elo, p_qf, p_sf, p_f, p_ch,
                                  ev, evtok, std, f"Q{quad}", draw_eff])
        print(f"**{title}**")
        if note:
            print(f"_{note}_")
        print(f"EV: {summary['gross_ev']:.2f}  |  "
              f"StdDev: {summary['portfolio_std']:.2f}  |  "
              f"Tokens: {summary['tokens']}/{summary['token_cap']}  |  "
              f"{winner_str}")
        print("```")
        print("\n".join(fixed_table(headers, display_rows)))
        print("```")
    else:
        print(f"**Total Portfolio EV:** {summary['gross_ev']:.2f} Points")
        print(f"**Portfolio StdDev:** {summary['portfolio_std']:.2f} Points")
        print(f"**Total Capital Spent:** {summary['tokens']} / {summary['token_cap']} Tokens")
        print(f"**{winner_str}**\n")
        print(f"| Selected Athlete | Gender | Cost | {elo_label} | QF% | SF% | F% | W% | EV | EV/Token | StdDev | Bracket Quadrant | DrawEff |")
        print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for row in rows:
            name, g, cost, elo, p_qf, p_sf, p_f, p_ch, ev, evtok, std, quad, draw_eff = row
            print(f"| **{name}** | {g} | {cost} | {elo} "
                  f"| {p_qf} | {p_sf} | {p_f} | {p_ch} "
                  f"| {ev} | {evtok} | {std} | Quarter {quad} | {draw_eff} |")


def print_draw_efficiency(rows, elo_label, cfg):
    headers = ["Player", "G", "Cost", elo_label,
               "NeutralEV", "EV", "Neut/Tok", "EV/Tok", "DrawEff"]
    if cfg.discord:
        print("**DRAW EFFICIENCY**")
        print("```")
        print("\n".join(fixed_table(headers, rows)))
        print("```")
    else:
        print("## DRAW EFFICIENCY\n")
        print(f"| Player | G | Cost | {elo_label} | NeutralEV | EV | Neut/Token | EV/Token | DrawEff |")
        print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for row in rows:
            name, g, cost, elo, n_ev, ev, n_evtok, evtok, draw_eff = row
            print(f"| **{name}** | {g} | {cost} | {elo} "
                  f"| {n_ev} | {ev} | {n_evtok} | {evtok} | {draw_eff} |")
        print()


def print_sim_comparison(rows, cfg):
    headers = ["Lineup", "E[EV]", "E[σ]", "Sim μ", "Sim σ", "Δμ"]
    if cfg.discord:
        print("**Analytical vs Simulated**")
        print("```")
        print("\n".join(fixed_table(headers, rows)))
        print("```")
    else:
        print("**Analytical vs Simulated**\n")
        print("| " + " | ".join(headers) + " |")
        print("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            print("| " + " | ".join(row) + " |")
        print()


def print_score_distributions(p_rows, ge_rows, headers, cfg):
    if cfg.discord:
        print("**Score Distribution  P(score = k)**")
        print("```")
        print("\n".join(fixed_table(headers, p_rows)))
        print("```")
        print("**Exceedance  P(score ≥ k)**")
        print("```")
        print("\n".join(fixed_table(headers, ge_rows)))
        print("```")
    else:
        print("### Score Distribution  P(score = k)\n")
        print("| " + " | ".join(headers) + " |")
        print("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in p_rows:
            print("| " + " | ".join(row) + " |")
        print()
        print("### Exceedance  P(score ≥ k)\n")
        print("| " + " | ".join(headers) + " |")
        print("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in ge_rows:
            print("| " + " | ".join(row) + " |")
        print()


def print_analysis(freq_rows, pair_rows, summary_str, cfg):
    print(f"\n**Analysis: {summary_str}**")
    if cfg.discord:
        print("**Player Frequency**")
        print("```")
        print("\n".join(fixed_table(["Player", "Count", "Freq%"], freq_rows)))
        print("```")
        if pair_rows:
            print("**Pairs (lift = observed / expected)**")
            print("```")
            print("\n".join(fixed_table(["Pair", "Count", "Lift"], pair_rows)))
            print("```")
    else:
        print("**Player Frequency**\n")
        print("| Player | Count | Freq% |")
        print("| --- | --- | --- |")
        for row in freq_rows:
            print("| " + " | ".join(row) + " |")
        if pair_rows:
            print("\n**Pairs (lift = observed / expected)**\n")
            print("| Pair | Count | Lift |")
            print("| --- | --- | --- |")
            for row in pair_rows:
                print("| " + " | ".join(row) + " |")


def print_best_player_at(rows, cfg):
    headers = ["Score", "P(≥k)", "EV", "Cost", "Player"]
    if cfg.discord:
        print("\n**Best single pick by P(score ≥ k)**")
        print("```")
        print("\n".join(fixed_table(headers, rows)))
        print("```")
    else:
        print("\n### Best single pick by P(score ≥ k)\n")
        print("| " + " | ".join(headers) + " |")
        print("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            print("| " + " | ".join(row) + " |")
        print()


def print_path(rows, headers, header_str, cfg):
    if cfg.discord:
        print(f"**{header_str}**")
        print("```")
        print("\n".join(fixed_table(headers, rows)))
        print("```")
    else:
        print(f"### {header_str}\n")
        print("| " + " | ".join(headers) + " |")
        print("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            print("| " + " | ".join(str(c) for c in row) + " |")
        print()


def print_path_simulations(rows1, p_rows, ge_rows, cfg):
    h1 = ["Round", "Sim", "Analytical", "Δ"]
    h2 = ["Score", "P(=k)"]
    h3 = ["Score", "P(≥k)"]
    if cfg.discord:
        print("\n**Simulated vs Analytical reach rates**")
        print("```")
        print("\n".join(fixed_table(h1, rows1)))
        print("```")
        print("**Score Distribution  P(score = k)**")
        print("```")
        print("\n".join(fixed_table(h2, p_rows)))
        print("```")
        print("**Exceedance  P(score ≥ k)**")
        print("```")
        print("\n".join(fixed_table(h3, ge_rows)))
        print("```")
    else:
        print("\n### Simulated vs Analytical reach rates\n")
        print("| " + " | ".join(h1) + " |")
        print("| " + " | ".join(["---"] * len(h1)) + " |")
        for row in rows1:
            print("| " + " | ".join(row) + " |")
        print()
        print("### Score Distribution  P(score = k)\n")
        print("| " + " | ".join(h2) + " |")
        print("| --- | --- |")
        for row in p_rows:
            print("| " + " | ".join(row) + " |")
        print()
        print("### Exceedance  P(score ≥ k)\n")
        print("| " + " | ".join(h3) + " |")
        print("| --- | --- |")
        for row in ge_rows:
            print("| " + " | ".join(row) + " |")
        print()


def print_best_at_table(rows, pool_size, cfg):
    header_line = f"**Best lineup by P(score ≥ k)  (across top-{pool_size} lineups)**"
    headers = ["Score", "P(≥k)", "EV", "Lineup"]
    if cfg.discord:
        print(header_line)
        print("```")
        print("\n".join(fixed_table(headers, rows)))
        print("```")
    else:
        print(f"{header_line}\n")
        print("| Score | P(≥k) | EV | Lineup |")
        print("| --- | --- | --- | --- |")
        for row in rows:
            print("| " + " | ".join(row) + " |")
        print()
