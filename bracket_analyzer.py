#!/usr/bin/env python3
import math
import random
from dataclasses import dataclass, field

from name_matching import _resolve_player
from elo_fetcher import update_elo_files
from data_loader import load_data as _load_data, load_preset_lineups as _load_preset_lineups
from bracket import (
    label_to_round as _label_to_round_fn,
    bracket_opponent_lines as _bracket_opponent_lines_fn,
    compute_ev as _compute_ev_fn,
    compute_draw_efficiency as _compute_draw_efficiency_fn,
)
from simulator import (
    simulate_tournament as _simulate_tournament_fn,
    score_lineup_from_sim as _score_lineup_from_sim_fn,
    cap_sim_pool as _cap_sim_pool_fn,
    run_simulations as _run_simulations_fn,
)
from optimizer import find_top_lineups as _find_top_lineups_fn
from scoring import ScoringModel
import rows as _rows
import output as _output
from output import OutputConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    elo_col: str = "elo"
    token_cap: int = 20
    discord: bool = True
    top_n: int = 1
    n_simulations: int = 0
    analyze: bool = False
    excluded: set = field(default_factory=set)
    included: set = field(default_factory=set)
    ev_floor: float = 0.0
    best_at: bool = False
    model: ScoringModel = field(default_factory=lambda: ScoringModel.from_final_rounds(3))
    lineups_path: str = None
    k_factor: float = 0
    lineup_size: int = None
    bo5: bool = False
    draw_efficiency: bool = False


# ---------------------------------------------------------------------------
# Mutation helpers (clear cache after mutating elo/advancements)
# ---------------------------------------------------------------------------

def apply_advancements(data, advancements_raw):
    """Parse Player:Round strings, resolve names, return {name: round_index} dict."""
    advancements = {}
    for entry in advancements_raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            print(f"ERROR: --advancements entry '{entry}' must be in Player:Round format")
            continue
        raw_name, _, raw_round = entry.rpartition(":")
        try:
            resolved = _resolve_player(raw_name.strip(), data.players)
            gender = data.players[resolved]["gender"]
            max_rounds = data.gender_max_rounds.get(gender, 7)
            rnd = _label_to_round_fn(raw_round.strip(), max_rounds)
            advancements[resolved] = rnd
            print(f"Advancing {resolved} through {raw_round.strip().upper()} (round {rnd})", flush=True)
        except ValueError as e:
            print(f"ERROR: {e}")
    data.section_cache.clear()
    return advancements


def apply_boost(data, boost_raw):
    """Parse Player:amount strings and mutate elo values in data.players."""
    for entry in boost_raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            print(f"ERROR: --boost entry '{entry}' must be in Player:amount format")
            continue
        raw_name, _, raw_amt = entry.rpartition(":")
        try:
            resolved = _resolve_player(raw_name.strip(), data.players)
            amt = float(raw_amt.strip())
            data.players[resolved]["elo"] += amt
            print(f"Boosting {resolved} by {amt:+.0f} → {data.players[resolved]['elo']:.0f}", flush=True)
        except ValueError as e:
            print(f"ERROR: {e}")
    data.section_cache.clear()


# ---------------------------------------------------------------------------
# EV computation
# ---------------------------------------------------------------------------

def _compute_player_evs(data, advancements, config):
    """Compute EV + draw efficiency for all players. Returns player_evs dict."""
    player_evs = {
        name: _compute_ev_fn(data, name, advancements, config.bo5, config.model)
        for name in data.players
    }
    for name in player_evs:
        if not data.players[name].get("is_priced") or data.players[name]["elo"] == 0:
            continue
        draw_eff, neutral_ev = _compute_draw_efficiency_fn(
            data, name, player_evs, config.model, config.bo5)
        player_evs[name]["draw_eff"] = draw_eff
        player_evs[name]["neutral_ev"] = neutral_ev
    return player_evs


# ---------------------------------------------------------------------------
# Optimizer wrapper (prints histogram as a side effect)
# ---------------------------------------------------------------------------

def _find_and_print_lineups(data, player_evs, n, config):
    """Run optimizer, print EV histogram, return (top_lineups, evaluated)."""
    top_lineups, evaluated, ev_histogram = _find_top_lineups_fn(
        data, player_evs, n,
        excluded=config.excluded,
        included=config.included,
        token_cap=config.token_cap,
        lineup_size=config.lineup_size,
        ev_floor=config.ev_floor,
    )
    ne = ev_histogram["n"]
    pct_vals = ev_histogram["percentiles"]
    buckets = ev_histogram["buckets"]
    pct_marks = [10, 25, 50, 75, 90]
    print(f"[optimizer] {ne:,} lineups evaluated  "
          f"min={ev_histogram['min']:.2f}  "
          + "  ".join(f"P{p}={pct_vals[p]:.2f}" for p in pct_marks)
          + f"  max={ev_histogram['max']:.2f}", flush=True)
    print("[optimizer] EV distribution:")
    max_count = max(buckets.values()) if buckets else 1
    for b in sorted(buckets):
        bar = "█" * int(buckets[b] / max_count * 40)
        print(f"  {b:5.1f}  {bar}  {buckets[b]}")
    return top_lineups, evaluated


# ---------------------------------------------------------------------------
# Orchestration: standard (top-N) path
# ---------------------------------------------------------------------------

def run_standard_optimization(data, player_evs, lineups, evaluated, title, note, config, advancements):
    """Print top-N lineup cards with optional simulation."""
    cfg = OutputConfig(discord=config.discord)
    sim_scores = None
    if config.n_simulations > 0:
        print(f"Running {config.n_simulations:,} simulations over {len(lineups):,} lineups...", flush=True)
        sim_scores = _run_simulations_fn(data, lineups, config.n_simulations,
                                         advancements, config.bo5, config.k_factor, config.model)
        print()

    for i, (ev, lineup) in enumerate(lineups):
        heading = title if config.top_n == 1 else f"{title} #{i + 1}"
        summary = _rows.lineup_summary(data, player_evs, lineup, config.model,
                                       config.token_cap, config.elo_col)
        lrows = _rows.lineup_rows(data, player_evs, lineup, config.model, config.elo_col)
        _output.print_lineup(lrows, summary, heading, note, cfg)
        if i < len(lineups) - 1:
            print()

    if sim_scores is not None:
        print()
        crows = _rows.sim_comparison_rows(data, lineups, sim_scores, player_evs, config.model)
        _output.print_sim_comparison(crows, cfg)
        print()
        p_rows, ge_rows, headers = _rows.score_distribution_rows(sim_scores)
        _output.print_score_distributions(p_rows, ge_rows, headers, cfg)

    if config.analyze:
        print()
        freq_rows, pair_rows, summary_str = _rows.analysis_rows(evaluated)
        _output.print_analysis(freq_rows, pair_rows, summary_str, cfg)


# ---------------------------------------------------------------------------
# Orchestration: best-at (simulation-first) path
# ---------------------------------------------------------------------------

def run_best_at_optimization(data, player_evs, evaluated, title, note, config,
                             advancements, preset_lineups=None):
    """Simulate top pool, show best lineup per P(score>=k) threshold."""
    cfg = OutputConfig(discord=config.discord)
    pool = _cap_sim_pool_fn(evaluated[-100:], config.n_simulations, "--best-at: ")
    print(f"Running {config.n_simulations:,} simulations over {len(pool):,} lineups for --best-at...",
          flush=True)
    pool_scores = _run_simulations_fn(data, pool, config.n_simulations,
                                      advancements, config.bo5, config.k_factor, config.model)
    print()

    n_trials = len(pool_scores[0])

    BEST_AT_THRESHOLDS = []
    threshold_winners = {}
    k = 2
    while True:
        ge_vals = [sum(1 for s in scores if s >= k) / n_trials * 100 for scores in pool_scores]
        best_idx = max(range(len(ge_vals)), key=lambda i: ge_vals[i])
        threshold_winners[k] = best_idx
        BEST_AT_THRESHOLDS.append(k)
        if ge_vals[best_idx] < 5.0:
            break
        k += 2

    def sim_floor(scores):
        n = len(scores)
        mean = sum(scores) / n
        std = math.sqrt(sum((s - mean) ** 2 for s in scores) / n)
        return mean - std

    floor_vals = [sim_floor(scores) for scores in pool_scores]
    floor_winner = max(range(len(floor_vals)), key=lambda i: floor_vals[i])

    pool_rank = {idx: n + 1 for n, idx in enumerate(
        sorted(range(len(pool)), key=lambda i: pool[i][0], reverse=True))}

    if preset_lineups is not None:
        unique_idxs = sorted(range(len(pool)), key=lambda i: pool_rank[i])
    else:
        unique_idxs = list(dict.fromkeys(
            [threshold_winners[k] for k in BEST_AT_THRESHOLDS] +
            [floor_winner] +
            [i for i in range(len(pool)) if pool_rank[i] <= 2]
        ))
        unique_idxs.sort(key=lambda i: pool_rank[i])
    seen = {idx: pool_rank[idx] for idx in unique_idxs}

    display_pool = [(seen[idx], pool[idx], pool_scores[idx])
                    for idx in sorted(seen, key=lambda i: seen[i])]
    display_lineups = [(ev, lineup) for _, (ev, lineup), _ in display_pool]
    display_scores  = [scores for _, _, scores in display_pool]
    display_labels  = [f"#{n_label}" for n_label, _, _ in display_pool]
    pool_label_map  = {idx: f"#{n}" for idx, n in seen.items()}

    for n_label, (ev, lineup), _ in display_pool:
        summary = _rows.lineup_summary(data, player_evs, lineup, config.model,
                                       config.token_cap, config.elo_col)
        lrows = _rows.lineup_rows(data, player_evs, lineup, config.model, config.elo_col)
        _output.print_lineup(lrows, summary, f"{title} #{n_label}", note, cfg)
        print()

    crows = _rows.sim_comparison_rows(data, display_lineups, display_scores,
                                      player_evs, config.model, display_labels)
    _output.print_sim_comparison(crows, cfg)
    print()
    p_rows, ge_rows, headers = _rows.score_distribution_rows(display_scores, display_labels)
    _output.print_score_distributions(p_rows, ge_rows, headers, cfg)
    print()

    rows = []
    for k in BEST_AT_THRESHOLDS:
        idx = threshold_winners[k]
        scores = pool_scores[idx]
        best_pct = sum(1 for s in scores if s >= k) / n_trials * 100
        best_ev, best_lineup = pool[idx]
        label = f"{pool_label_map[idx]}: {', '.join(p.split()[-1] for p in best_lineup)}"
        rows.append([f"≥{k}", f"{best_pct:.1f}%", f"{best_ev:.2f}", label])

    f_scores = pool_scores[floor_winner]
    f_mean = sum(f_scores) / n_trials
    f_std = math.sqrt(sum((s - f_mean) ** 2 for s in f_scores) / n_trials)
    f_ev, f_lineup = pool[floor_winner]
    rows.append([
        "μ−σ", f"{f_mean - f_std:.2f}", f"{f_ev:.2f}",
        f"{pool_label_map[floor_winner]}: {', '.join(p.split()[-1] for p in f_lineup)}",
    ])

    _output.print_best_at_table(rows, len(pool), cfg)

    if config.analyze:
        print()
        freq_rows, pair_rows, summary_str = _rows.analysis_rows(evaluated)
        _output.print_analysis(freq_rows, pair_rows, summary_str, cfg)


# ---------------------------------------------------------------------------
# Orchestration: path analysis
# ---------------------------------------------------------------------------

def run_path(data, player_name, config, advancements):
    """Print bracket path table and optional path simulations."""
    cfg = OutputConfig(discord=config.discord)
    ev = _compute_ev_fn(data, player_name, advancements, config.bo5, config.model)
    prows, headers, header_str = _rows.path_rows(
        data, player_name, ev, advancements, config.bo5, config.model, config.elo_col)
    _output.print_path(prows, headers, header_str, cfg)

    n_sims = config.n_simulations
    if n_sims > 0:
        p_data = data.players[player_name]
        gender = p_data["gender"]
        max_rounds = data.gender_max_rounds.get(gender, 7)
        size = 2 ** max_rounds
        opp_sections = _bracket_opponent_lines_fn(p_data["line"], size)

        rounds_reached = [0] * (max_rounds + 1)
        score_counts = {}

        print(f"Running {n_sims:,} simulations for {player_name}...", flush=True)
        rng = random.Random()
        for _ in range(n_sims):
            live_elos = ({n: p["elo"] for n, p in data.players.items()}
                         if config.k_factor else None)
            result = _simulate_tournament_fn(data, gender, rng, advancements,
                                             config.bo5, config.k_factor, live_elos)
            r = result.get(player_name, 0)
            for rnd in range(1, r + 1):
                rounds_reached[rnd] += 1
            score = _score_lineup_from_sim_fn(data, (player_name,), result, config.model)
            score_counts[score] = score_counts.get(score, 0) + 1

        print()
        rows1, p_rows, ge_rows = _rows.path_sim_rows(
            player_name, rounds_reached, score_counts, ev["all_probs"], n_sims,
            data, opp_sections, advancements, config.bo5,
        )
        _output.print_path_simulations(rows1, p_rows, ge_rows, cfg)


# ---------------------------------------------------------------------------
# Orchestration: full optimization run
# ---------------------------------------------------------------------------

def run_optimization(data, config, advancements):
    """Compute EVs, print player pool, find and print lineups."""
    player_evs = _compute_player_evs(data, advancements, config)
    elo_label = {"elo": "Elo", "gelo": "gElo", "celo": "cElo", "helo": "hElo"}.get(
        config.elo_col, config.elo_col)
    cfg = OutputConfig(discord=config.discord)

    print("**PLAYER POOL**\n" if config.discord else "## PLAYER POOL\n")

    genders_present = {pd["gender"] for name, pd in data.players.items()
                       if not name.startswith("__BYE")}
    priced = [n for n, p in data.players.items() if p["is_priced"]]
    top_evtok = set(sorted(priced,
                           key=lambda n: player_evs[n]["ev"] / data.players[n]["cost"],
                           reverse=True)[:5])

    if len(genders_present) > 1:
        for gender, title in (("M", "Men"), ("F", "Women")):
            prows = _rows.pool_section_rows(data, player_evs, gender, config.elo_col, top_evtok)
            _output.print_pool_section(prows, title, elo_label, cfg)
    else:
        gender = next(iter(genders_present))
        prows = _rows.pool_section_rows(data, player_evs, gender, config.elo_col, top_evtok)
        _output.print_pool_section(prows, None, elo_label, cfg)

    if config.draw_efficiency:
        print("\n---\n")
        drows = _rows.draw_efficiency_rows(data, player_evs, config.elo_col)
        _output.print_draw_efficiency(drows, elo_label, cfg)

    print("\n---\n")

    no_costs = not any(pd["is_priced"] for pd in data.players.values()
                       if not any(True for _ in []))
    # no-cost mode: is_priced is false for all if no costs file was given
    all_priced = [n for n, p in data.players.items() if p["is_priced"]]
    no_costs = not all_priced and config.lineup_size is None and not config.lineups_path

    if no_costs:
        if config.n_simulations > 0:
            _run_best_player_at(data, player_evs, config, advancements)
        return

    if config.lineups_path:
        preset = _load_preset_lineups(config.lineups_path, data, player_evs)
        if not preset:
            print("ERROR: no valid lineups found in lineups file.")
            return
        evaluated = sorted(preset, key=lambda x: x[0])
        lineups = preset
        n_display = len(preset)
        if not config.discord:
            print(f"## PRESET LINEUPS ({n_display})")
        title = "LINEUP"
    else:
        n_display = config.top_n
        if not config.discord:
            print("## OPTIMAL LINEUP" if n_display == 1 else "## TOP LINEUPS")
        title = "OPTIMAL LINEUP" if n_display == 1 else "LINEUP"
        lineups, evaluated = _find_and_print_lineups(data, player_evs, n_display, config)

    if config.best_at and config.n_simulations > 0:
        run_best_at_optimization(data, player_evs, evaluated, title, None, config,
                                 advancements, preset_lineups=(config.lineups_path is not None))
    else:
        run_standard_optimization(data, player_evs, lineups, evaluated, title, None,
                                  config, advancements)


def _run_best_player_at(data, player_evs, config, advancements):
    """Simulate individual players, print best pick per P(score>=k) threshold."""
    rng = random.Random()
    priced = [n for n, p in data.players.items() if p["is_priced"]]
    scores = {name: [] for name in priced}

    print(f"Running {config.n_simulations:,} simulations over {len(priced):,} players...", flush=True)
    for _ in range(config.n_simulations):
        live_elos = ({n: p["elo"] for n, p in data.players.items()}
                     if config.k_factor else None)
        m_result = _simulate_tournament_fn(data, "M", rng, advancements, config.bo5,
                                           config.k_factor, live_elos)
        f_result = _simulate_tournament_fn(data, "F", rng, advancements, config.bo5,
                                           config.k_factor, live_elos)
        combined = {**m_result, **f_result}
        for name in priced:
            scores[name].append(_score_lineup_from_sim_fn(data, (name,), combined, config.model))

    cfg = OutputConfig(discord=config.discord)
    brows = _rows.best_player_at_rows(priced, scores, player_evs, data, config.model)
    _output.print_best_player_at(brows, cfg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        usage="%(prog)s -b BRACKET [-p COSTS] [-e ELO | -m MEN -w WOMEN] [options]"
    )
    parser.add_argument("-p", dest="costs_path", default=None)
    parser.add_argument("-d", "-b", dest="draw_path", default=None, metavar="FILE",
                        help="(required) Bracket/draw CSV")
    parser.add_argument("-e", dest="elo_path", default=None, help="Gender-neutral Elo CSV")
    parser.add_argument("-m", dest="men_path", default="atp_elo.csv",
                        help="Men's Elo CSV (default: atp_elo.csv)")
    parser.add_argument("-w", dest="women_path", default="wta_elo.csv",
                        help="Women's Elo CSV (default: wta_elo.csv)")
    surface = parser.add_mutually_exclusive_group()
    surface.add_argument("--grass", dest="elo_col", action="store_const", const="gelo",
                         help="Use grass-court Elo")
    surface.add_argument("--clay",  dest="elo_col", action="store_const", const="celo",
                         help="Use clay-court Elo")
    surface.add_argument("--hard",  dest="elo_col", action="store_const", const="helo",
                         help="Use hard-court Elo")
    parser.set_defaults(elo_col="elo")
    parser.add_argument("--update-elo", action="store_true",
                        help="Fetch latest Elo ratings from tennisabstract.com and update atp_elo.csv / wta_elo.csv")
    parser.add_argument("--markdown", action="store_true",
                        help="Format output as Markdown tables instead of Discord code blocks")
    parser.add_argument("--path", dest="path_player", default=None, metavar="PLAYER",
                        help="Show bracket path analysis for a single player")
    parser.add_argument("--top", dest="top_n", type=int, default=1, metavar="N",
                        help="Show top N lineups by EV (default: 1)")
    parser.add_argument("--simulate", dest="n_simulations", type=int, default=0, metavar="N",
                        help="Run N Monte Carlo simulations (e.g. --simulate 10000)")
    parser.add_argument("--analyze", action="store_true",
                        help="Show player frequency and pair co-occurrence for top EV buckets")
    parser.add_argument("--exclude", dest="exclude_raw", default=None, metavar="PLAYERS",
                        help="Comma-separated players to exclude from lineups (fuzzy matched)")
    parser.add_argument("--include", dest="include_raw", default=None, metavar="PLAYERS",
                        help="Comma-separated players to force into every lineup (fuzzy matched)")
    parser.add_argument("--ev-floor", dest="ev_floor", type=float, default=0.5, metavar="N",
                        help="Evaluate all lineups within N EV points of optimal (default: 0.5)")
    parser.add_argument("--best-at", dest="best_at", action="store_true",
                        help="Show best lineup by P(score >= k) at each scoring threshold")
    parser.add_argument("--scoring-rounds", dest="scoring_rounds", type=int, default=3, metavar="N",
                        help="Award points for reaching the final N rounds + winning (default: 3)")
    parser.add_argument("--tokens", dest="token_cap", type=int, default=20, metavar="N",
                        help="Token budget cap (default: 20)")
    parser.add_argument("--size", dest="lineup_size", type=int, default=None, metavar="N",
                        help="Pick exactly N players with no token constraint")
    parser.add_argument("--lineups", dest="lineups_path", default=None, metavar="FILE",
                        help="Text file of preset lineups to evaluate (one per line, comma-separated)")
    parser.add_argument("--k-factor", dest="k_factor", type=float, default=0, metavar="K",
                        help="Elo K-factor for live simulation updates (0 = disabled)")
    parser.add_argument("--bo5", dest="bo5", action="store_true",
                        help="Adjust win probabilities for best-of-five matches")
    parser.add_argument("--draw-efficiency", dest="draw_efficiency", action="store_true",
                        help="Print draw efficiency table (actual EV / expected EV)")
    parser.add_argument("--advancements", dest="advancements_raw", default=None,
                        metavar="PLAYER:ROUND,...",
                        help="Force players through a round, e.g. \"Djokovic:QF,Sinner:F\"")
    parser.add_argument("--boost", dest="boost_raw", default=None, metavar="PLAYER:AMT,...",
                        help="Elo boosts, e.g. \"Sinner:50,Djokovic:-25\"")
    args = parser.parse_args()

    if not args.update_elo and not args.draw_path:
        parser.print_usage()
        raise SystemExit(1)

    if args.update_elo:
        update_elo_files()
        raise SystemExit(0)

    data = _load_data(
        draw_path=args.draw_path,
        costs_path=args.costs_path,
        elo_path=args.elo_path,
        men_path=args.men_path,
        women_path=args.women_path,
        elo_col=args.elo_col,
    )

    advancements = {}
    if args.advancements_raw:
        advancements = apply_advancements(data, args.advancements_raw)

    if args.boost_raw:
        apply_boost(data, args.boost_raw)

    excluded = set()
    if args.exclude_raw:
        for raw in args.exclude_raw.split(","):
            raw = raw.strip()
            if not raw:
                continue
            try:
                resolved = _resolve_player(raw, data.players)
                excluded.add(resolved)
                print(f"Excluding: {resolved}", flush=True)
            except ValueError as e:
                print(f"ERROR: {e}")

    included = set()
    if args.include_raw:
        for raw in args.include_raw.split(","):
            raw = raw.strip()
            if not raw:
                continue
            try:
                resolved = _resolve_player(raw, data.players)
                if not data.players[resolved]["is_priced"]:
                    print(f"ERROR: {resolved} is not a priced player and cannot be included")
                    continue
                included.add(resolved)
                print(f"Including: {resolved}", flush=True)
            except ValueError as e:
                print(f"ERROR: {e}")

    config = Config(
        elo_col=args.elo_col,
        token_cap=args.token_cap,
        discord=not args.markdown,
        top_n=args.top_n,
        n_simulations=args.n_simulations or (10000 if args.best_at else 0),
        analyze=args.analyze,
        excluded=excluded,
        included=included,
        ev_floor=args.ev_floor,
        best_at=args.best_at,
        model=ScoringModel.from_final_rounds(args.scoring_rounds),
        lineups_path=args.lineups_path,
        k_factor=args.k_factor,
        lineup_size=args.lineup_size,
        bo5=args.bo5,
        draw_efficiency=args.draw_efficiency,
    )

    if args.path_player:
        ignored = [f"--{f}" for f, v in [
            ("top",      args.top_n != 1),
            ("analyze",  args.analyze),
            ("best-at",  args.best_at),
            ("ev-floor", args.ev_floor != 0.5),
            ("exclude",  bool(args.exclude_raw)),
            ("include",  bool(args.include_raw)),
            ("lineups",  bool(args.lineups_path)),
        ] if v]
        if ignored:
            print(f"WARNING: --path ignores {', '.join(ignored)}", flush=True)
        try:
            resolved = _resolve_player(args.path_player, data.players)
            run_path(data, resolved, config, advancements)
        except ValueError as e:
            print(f"ERROR: {e}")
    else:
        run_optimization(data, config, advancements)
