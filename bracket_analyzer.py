#!/usr/bin/env python3
import csv
import math

from name_matching import (
    _normalize,
    _build_norm_index,
    _lookup_normalized,
    _fuzzy_matches,
    _resolve_player,
)
from elo_math import calculate_match_win_prob
from elo_fetcher import update_elo_files
from data_loader import BracketData, load_data as _load_data, load_preset_lineups as _load_preset_lineups
from bracket import (
    label_to_round as _label_to_round_fn,
    round_label as _round_label_fn,
    bracket_opponent_lines as _bracket_opponent_lines_fn,
    section_win_probs as _section_win_probs_fn,
    expected_win_prob as _expected_win_prob_fn,
    compute_ev as _compute_ev_fn,
    compute_draw_efficiency as _compute_draw_efficiency_fn,
)
from simulator import (
    simulate_tournament as _simulate_tournament_fn,
    score_lineup_from_sim as _score_lineup_from_sim_fn,
    cap_sim_pool as _cap_sim_pool_fn,
    run_simulations as _run_simulations_fn,
)
from optimizer import (
    meeting_block_size as _meeting_block_size_fn,
    pairwise_cov as _pairwise_cov_fn,
    score_variance as _score_variance_fn,
    lineup_variance as _lineup_variance_fn,
    max_lineup_score as _max_lineup_score_fn,
    find_top_lineups as _find_top_lineups_fn,
)
from scoring import ScoringModel
import rows as _rows
import output as _output
from output import OutputConfig as _OutputConfig



class ComprehensiveFantasyOptimizer:

    def __init__(
        self,
        costs_path=None,
        draw_path=None,
        men_path=None,
        women_path=None,
        elo_path=None,
        elo_col="elo",
        token_cap=20,
        discord=True,
        top_n=1,
        n_simulations=0,
        analyze=False,
        excluded=None,
        included=None,
        ev_floor=0.0,
        best_at=False,
        scoring_rounds=3,
        lineups_path=None,
        k_factor=0,
        lineup_size=None,
        bo5=False,
        advancements=None,
        draw_efficiency=False,
    ):
        self.costs_path = costs_path
        self.draw_path = draw_path
        self.men_path = men_path
        self.women_path = women_path
        self.elo_path = elo_path
        self.elo_col = elo_col
        self.token_cap = token_cap
        self.discord = discord
        self.top_n = top_n
        self.n_simulations = n_simulations
        self.analyze = analyze
        self.excluded = set(excluded) if excluded else set()
        self.included = set(included) if included else set()
        self.ev_floor = ev_floor
        self.best_at = best_at
        self.scoring_rounds = scoring_rounds
        self._model = ScoringModel.from_final_rounds(scoring_rounds)
        self.lineups_path = lineups_path
        self.k_factor = k_factor
        self.lineup_size = lineup_size
        self.bo5 = bo5
        self.advancements = advancements or {}
        self.draw_efficiency = draw_efficiency
        self._data = None  # BracketData, set by load_data()

    def _label_to_round(self, label, max_rounds):
        return _label_to_round_fn(label, max_rounds)

    def _round_label(self, rnd, max_rounds):
        return _round_label_fn(rnd, max_rounds)

    def _max_rounds(self, gender):
        return self._data.gender_max_rounds.get(gender, 7)

    def load_data(self):
        self._data = _load_data(
            draw_path=self.draw_path,
            costs_path=self.costs_path,
            elo_path=self.elo_path,
            men_path=self.men_path,
            women_path=self.women_path,
            elo_col=self.elo_col,
        )
        # Keep legacy aliases for methods not yet migrated
        self.players = self._data.players
        self._gender_max_rounds = self._data.gender_max_rounds
        self._line_index = self._data.line_index
        self._line_to_name = self._data.line_to_name
        self._section_cache = self._data.section_cache

    def _win_prob(self, elo_a, elo_b, gender=None):
        return calculate_match_win_prob(elo_a, elo_b, bo5=self.bo5, gender=gender)

    def _bracket_opponent_lines(self, line, size=128):
        return _bracket_opponent_lines_fn(line, size)

    def _section_win_probs(self, lines, gender, first_round=1):
        return _section_win_probs_fn(self._data, lines, gender, self.advancements, self.bo5, first_round)

    def _expected_win_prob(self, player_elo, lines, gender, fallback_elo, first_round=1, facing_round=None):
        return _expected_win_prob_fn(self._data, player_elo, lines, gender, self.advancements,
                                     self.bo5, fallback_elo, first_round, facing_round)

    def compute_ev(self, player_name):
        return _compute_ev_fn(self._data, player_name, self.advancements, self.bo5, self._model)

    def compute_draw_efficiency(self, player_name, player_evs):
        return _compute_draw_efficiency_fn(self._data, player_name, player_evs, self._model, self.bo5)

    def _meeting_block_size(self, name_a, name_b):
        return _meeting_block_size_fn(self._data, name_a, name_b)

    def _pairwise_cov(self, name_a, name_b, evs):
        return _pairwise_cov_fn(self._data, name_a, name_b, evs, self._model)

    def _score_variance(self, name, evs):
        return _score_variance_fn(self._data, name, evs, self._model)

    def _lineup_variance(self, lineup, evs):
        return _lineup_variance_fn(self._data, lineup, evs, self._model)

    def _max_lineup_score(self, lineup):
        return _max_lineup_score_fn(self._data, lineup, self._model)

    def _print_lineup(self, title, note, player_evs, best_lineup):
        cfg = _OutputConfig(discord=self.discord)
        summary = _rows.lineup_summary(self._data, player_evs, best_lineup, self._model,
                                       self.token_cap, self.elo_col)
        lrows = _rows.lineup_rows(self._data, player_evs, best_lineup, self._model, self.elo_col)
        _output.print_lineup(lrows, summary, title, note, cfg)

    def _simulate_tournament(self, gender, rng, live_elos=None):
        return _simulate_tournament_fn(self._data, gender, rng, self.advancements, self.bo5,
                                       self.k_factor, live_elos)

    def _score_lineup_from_sim(self, lineup, rounds_won):
        return _score_lineup_from_sim_fn(self._data, lineup, rounds_won, self._model)

    def _cap_sim_pool(self, pool, n_trials, label=""):
        return _cap_sim_pool_fn(pool, n_trials, label)

    def run_simulations(self, lineups, n_trials=10000):
        return _run_simulations_fn(self._data, lineups, n_trials, self.advancements,
                                   self.bo5, self.k_factor, self._model)

    def _find_top_lineups(self, player_evs, n):
        """Return the top-n distinct lineups by gross EV using branch-and-bound DFS."""
        top_lineups, evaluated, ev_histogram = _find_top_lineups_fn(
            self._data, player_evs, n,
            excluded=self.excluded,
            included=self.included,
            token_cap=self.token_cap,
            lineup_size=self.lineup_size,
            ev_floor=self.ev_floor,
        )
        ne = ev_histogram["n"]
        pct_vals = ev_histogram["percentiles"]
        buckets = ev_histogram["buckets"]
        pct = [10, 25, 50, 75, 90]
        print(f"[optimizer] {ne:,} lineups evaluated  "
              f"min={ev_histogram['min']:.2f}  "
              + "  ".join(f"P{p}={pct_vals[p]:.2f}" for p in pct)
              + f"  max={ev_histogram['max']:.2f}", flush=True)
        print("[optimizer] EV distribution:")
        max_count = max(buckets.values()) if buckets else 1
        for b in sorted(buckets):
            bar = "█" * int(buckets[b] / max_count * 40)
            print(f"  {b:5.1f}  {bar}  {buckets[b]}")
        return top_lineups, evaluated

    def _print_sim_comparison(self, lineups, sim_scores, player_evs, labels=None):
        cfg = _OutputConfig(discord=self.discord)
        crows = _rows.sim_comparison_rows(self._data, lineups, sim_scores, player_evs, self._model, labels)
        _output.print_sim_comparison(crows, cfg)

    def _print_score_distributions(self, lineups, sim_scores, labels=None):
        cfg = _OutputConfig(discord=self.discord)
        p_rows, ge_rows, headers = _rows.score_distribution_rows(sim_scores, labels)
        _output.print_score_distributions(p_rows, ge_rows, headers, cfg)

    def _print_analysis(self, evaluated, top_k=100):
        if not evaluated:
            return
        cfg = _OutputConfig(discord=self.discord)
        freq_rows, pair_rows, summary_str = _rows.analysis_rows(evaluated, top_k)
        _output.print_analysis(freq_rows, pair_rows, summary_str, cfg)

    def _print_best_player_at(self, player_evs):
        import random
        n_trials = self.n_simulations
        rng = random.Random()
        priced = [n for n, p in self.players.items() if p["is_priced"]]
        scores = {name: [] for name in priced}

        print(f"Running {n_trials:,} simulations over {len(priced):,} players...", flush=True)
        for _ in range(n_trials):
            m_result = self._simulate_tournament("M", rng,
                dict({n: p["elo"] for n, p in self.players.items()}) if self.k_factor else None)
            f_result = self._simulate_tournament("F", rng,
                dict({n: p["elo"] for n, p in self.players.items()}) if self.k_factor else None)
            combined = {**m_result, **f_result}
            for name in priced:
                scores[name].append(self._score_lineup_from_sim((name,), combined))

        cfg = _OutputConfig(discord=self.discord)
        brows = _rows.best_player_at_rows(priced, scores, player_evs, self._data, self._model)
        _output.print_best_player_at(brows, cfg)

    def _print_path_simulations(self, player_name, n_trials):
        import random
        rng = random.Random()
        pd = self.players[player_name]
        gender = pd["gender"]
        max_rounds = self._max_rounds(gender)
        size = 2 ** max_rounds
        opp_sections = _bracket_opponent_lines_fn(pd["line"], size)

        rounds_reached = [0] * (max_rounds + 1)
        score_counts = {}

        print(f"Running {n_trials:,} simulations for {player_name}...", flush=True)
        for _ in range(n_trials):
            live_elos = {n: p["elo"] for n, p in self.players.items()} if self.k_factor else None
            result = self._simulate_tournament(gender, rng, live_elos)
            r = result.get(player_name, 0)
            for rnd in range(1, r + 1):
                rounds_reached[rnd] += 1
            score = self._score_lineup_from_sim((player_name,), result)
            score_counts[score] = score_counts.get(score, 0) + 1

        ev = self.compute_ev(player_name)
        rows1, p_rows, ge_rows = _rows.path_sim_rows(
            player_name, rounds_reached, score_counts, ev["all_probs"], n_trials,
            self._data, opp_sections, self.advancements, self.bo5,
        )
        cfg = _OutputConfig(discord=self.discord)
        _output.print_path_simulations(rows1, p_rows, ge_rows, cfg)

    def load_preset_lineups(self, player_evs):
        return _load_preset_lineups(self.lineups_path, self._data, player_evs)

    def _optimize_and_print(self, title, note, player_evs, top_n=1, n_simulations=0, analyze=False, preset_lineups=None):
        if preset_lineups is not None:
            lineups = preset_lineups
            evaluated = sorted(lineups, key=lambda x: x[0])
        else:
            lineups, evaluated = self._find_top_lineups(player_evs, top_n)

        if self.best_at and n_simulations > 0:
            # Simulate top-50 pool, identify winners per threshold, deduplicate
            pool = self._cap_sim_pool(evaluated[-100:], n_simulations, "--best-at: ")
            print(f"Running {n_simulations:,} simulations over {len(pool):,} lineups for --best-at...", flush=True)
            pool_scores = self.run_simulations(pool, n_simulations)
            print()

            n_trials = len(pool_scores[0])

            # Build thresholds dynamically: keep adding until best P(≥k) drops below 5%
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

            # Best floor: lineup with highest (mean - stddev)
            def sim_floor(scores):
                n = len(scores)
                mean = sum(scores) / n
                std = math.sqrt(sum((s - mean) ** 2 for s in scores) / n)
                return mean - std

            floor_vals = [sim_floor(scores) for scores in pool_scores]
            floor_winner = max(range(len(floor_vals)), key=lambda i: floor_vals[i])

            # Assign pool rank by EV descending (#1 = highest EV in pool)
            pool_rank = {idx: n + 1 for n, idx in enumerate(sorted(range(len(pool)), key=lambda i: pool[i][0], reverse=True))}
            # With preset lineups show all; otherwise deduplicate to threshold winners + top-2 + floor
            if preset_lineups is not None:
                unique_idxs = list(range(len(pool)))
                unique_idxs.sort(key=lambda i: pool_rank[i])
            else:
                unique_idxs = list(dict.fromkeys(
                    [threshold_winners[k] for k in BEST_AT_THRESHOLDS] +
                    [floor_winner] +
                    [i for i in range(len(pool)) if pool_rank[i] <= 2]
                ))
                unique_idxs.sort(key=lambda i: pool_rank[i])
            seen = {idx: pool_rank[idx] for idx in unique_idxs}

            # Build display lineups and scores in #N order
            display_pool = [(seen[idx], pool[idx], pool_scores[idx]) for idx in sorted(seen, key=lambda i: seen[i])]
            display_lineups = [(ev, lineup) for _, (ev, lineup), _ in display_pool]
            display_scores = [scores for _, _, scores in display_pool]
            display_labels = [f"#{n_label}" for n_label, _, _ in display_pool]
            # Map pool index → #N label for best-at table
            pool_label_map = {idx: f"#{n}" for idx, n in seen.items()}

            # Print individual lineup cards
            for n_label, (ev, lineup), _ in display_pool:
                heading = f"{title} #{n_label}"
                self._print_lineup(heading, note, player_evs, lineup)
                print()

            self._print_sim_comparison(display_lineups, display_scores, player_evs, display_labels)
            print()
            self._print_score_distributions(display_lineups, display_scores, display_labels)
            print()

            # Best-at table using #N labels
            rows = []
            for k in BEST_AT_THRESHOLDS:
                idx = threshold_winners[k]
                scores = pool_scores[idx]
                best_pct = sum(1 for s in scores if s >= k) / n_trials * 100
                best_ev, best_lineup = pool[idx]
                last_names = [p.split()[-1] for p in best_lineup]
                label = f"{pool_label_map[idx]}: {', '.join(last_names)}"
                rows.append([f"≥{k}", f"{best_pct:.1f}%", f"{best_ev:.2f}", label])

            # Best floor row (μ - σ)
            f_scores = pool_scores[floor_winner]
            f_mean = sum(f_scores) / n_trials
            f_std = math.sqrt(sum((s - f_mean) ** 2 for s in f_scores) / n_trials)
            f_ev, f_lineup = pool[floor_winner]
            f_last_names = [p.split()[-1] for p in f_lineup]
            rows.append([
                "μ−σ",
                f"{f_mean - f_std:.2f}",
                f"{f_ev:.2f}",
                f"{pool_label_map[floor_winner]}: {', '.join(f_last_names)}",
            ])

            cfg = _OutputConfig(discord=self.discord)
            _output.print_best_at_table(rows, len(pool), cfg)

        else:
            sim_scores = None
            if n_simulations > 0:
                print(f"Running {n_simulations:,} simulations over {len(lineups):,} lineups...", flush=True)
                sim_scores = self.run_simulations(lineups, n_simulations)
                print()

            for i, (ev, lineup) in enumerate(lineups):
                heading = title if top_n == 1 else f"{title} #{i + 1}"
                self._print_lineup(heading, note, player_evs, lineup)
                if i < len(lineups) - 1:
                    print()

            if sim_scores is not None:
                print()
                self._print_sim_comparison(lineups, sim_scores, player_evs)
                print()
                self._print_score_distributions(lineups, sim_scores)

        if analyze:
            print()
            self._print_analysis(evaluated)

    def _print_pool_section(self, title, gender, player_evs, elo_label, top_evtok_names=None):
        cfg = _OutputConfig(discord=self.discord)
        prows = _rows.pool_section_rows(self._data, player_evs, gender, self.elo_col, top_evtok_names)
        _output.print_pool_section(prows, title, elo_label, cfg)

    def print_path(self, player_name):
        cfg = _OutputConfig(discord=self.discord)
        ev = self.compute_ev(player_name)
        prows, headers, header_str = _rows.path_rows(
            self._data, player_name, ev, self.advancements, self.bo5, self._model, self.elo_col)
        _output.print_path(prows, headers, header_str, cfg)

    def _print_draw_efficiency(self, player_evs, elo_label):
        cfg = _OutputConfig(discord=self.discord)
        drows = _rows.draw_efficiency_rows(self._data, player_evs, self.elo_col)
        _output.print_draw_efficiency(drows, elo_label, cfg)

    def run_optimization(self):
        if not self.players:
            self.load_data()
        player_evs = {name: self.compute_ev(name) for name in self.players}
        for name in player_evs:
            if not self.players[name].get("is_priced") or self.players[name]["elo"] == 0:
                continue
            draw_eff, neutral_ev = self.compute_draw_efficiency(name, player_evs)
            player_evs[name]["draw_eff"] = draw_eff
            player_evs[name]["neutral_ev"] = neutral_ev
        elo_label = {"elo": "Elo", "gelo": "gElo", "celo": "cElo", "helo": "hElo"}.get(self.elo_col, self.elo_col)

        if self.discord:
            print("**PLAYER POOL**\n")
        else:
            print("## PLAYER POOL\n")

        genders_present = {pd["gender"] for name, pd in self.players.items() if not name.startswith("__BYE")}
        priced = [n for n, p in self.players.items() if p["is_priced"]]
        top_evtok = set(sorted(priced, key=lambda n: player_evs[n]["ev"] / self.players[n]["cost"], reverse=True)[:5])
        if len(genders_present) > 1:
            self._print_pool_section("Men", "M", player_evs, elo_label, top_evtok)
            self._print_pool_section("Women", "F", player_evs, elo_label, top_evtok)
        else:
            self._print_pool_section(None, next(iter(genders_present)), player_evs, elo_label, top_evtok)

        if self.draw_efficiency:
            print("\n---\n")
            self._print_draw_efficiency(player_evs, elo_label)

        print("\n---\n")

        preset = None
        no_costs = not self.costs_path
        if no_costs and self.lineup_size is None and not self.lineups_path:
            if self.n_simulations > 0:
                self._print_best_player_at(player_evs)
            return

        if self.lineups_path:
            preset = self.load_preset_lineups(player_evs)
            if not preset:
                print("ERROR: no valid lineups found in lineups file.")
                return
            n_display = len(preset)
            if not self.discord:
                print(f"## PRESET LINEUPS ({n_display})")
            title = "LINEUP"
        else:
            n_display = self.top_n
            if not self.discord:
                print("## OPTIMAL LINEUP" if n_display == 1 else "## TOP LINEUPS")
            title = "OPTIMAL LINEUP" if n_display == 1 else "LINEUP"

        self._optimize_and_print(
            title,
            None,
            player_evs,
            top_n=n_display,
            n_simulations=self.n_simulations,
            analyze=self.analyze,
            preset_lineups=preset,
        )



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        usage="%(prog)s -b BRACKET [-p COSTS] [-e ELO | -m MEN -w WOMEN] [options]"
    )
    parser.add_argument("-p", dest="costs_path", default=None)
    parser.add_argument("-d", "-b", dest="draw_path", default=None, metavar="FILE",
                        help="(required) Bracket/draw CSV")
    parser.add_argument("-e", dest="elo_path", default=None, help="Gender-neutral Elo CSV")
    parser.add_argument("-m", dest="men_path", default="atp_elo.csv", help="Men's Elo CSV (default: atp_elo.csv)")
    parser.add_argument("-w", dest="women_path", default="wta_elo.csv", help="Women's Elo CSV (default: wta_elo.csv)")
    surface = parser.add_mutually_exclusive_group()
    surface.add_argument("--grass", dest="elo_col", action="store_const", const="gelo", help="Use grass-court Elo")
    surface.add_argument("--clay", dest="elo_col", action="store_const", const="celo", help="Use clay-court Elo")
    surface.add_argument("--hard", dest="elo_col", action="store_const", const="helo", help="Use hard-court Elo")
    parser.set_defaults(elo_col="elo")
    parser.add_argument("--update-elo", action="store_true",
                        help="Fetch latest Elo ratings from tennisabstract.com and update atp_elo.csv / wta_elo.csv")
    parser.add_argument("--markdown", action="store_true",
                        help="Format output as Markdown tables instead of Discord code blocks")
    parser.add_argument("--path", dest="path_player", default=None, metavar="PLAYER",
                        help="Show bracket path analysis for a single player instead of running optimization")
    parser.add_argument("--top", dest="top_n", type=int, default=1, metavar="N",
                        help="Show top N lineups by EV (default: 1)")
    parser.add_argument("--simulate", dest="n_simulations", type=int, default=0, metavar="N",
                        help="Run N Monte Carlo tournament simulations and show score percentiles (e.g. --simulate 10000)")
    parser.add_argument("--analyze", action="store_true",
                        help="Show player frequency and pair co-occurrence for top EV buckets")
    parser.add_argument("--exclude", dest="exclude_raw", default=None, metavar="PLAYERS",
                        help="Comma-separated players to exclude from lineups (fuzzy matched)")
    parser.add_argument("--include", dest="include_raw", default=None, metavar="PLAYERS",
                        help="Comma-separated players to force into every lineup (fuzzy matched)")
    parser.add_argument("--ev-floor", dest="ev_floor", type=float, default=0.5, metavar="N",
                        help="Evaluate all lineups within N EV points of optimal (e.g. --ev-floor 1.0)")
    parser.add_argument("--best-at", dest="best_at", action="store_true",
                        help="Show best lineup by P(score >= k) at each scoring threshold")
    parser.add_argument("--scoring-rounds", dest="scoring_rounds", type=int, default=3, metavar="N",
                        help="Award points for reaching the final N rounds + winning (default: 3)")
    parser.add_argument("--tokens", dest="token_cap", type=int, default=20, metavar="N",
                        help="Token budget cap for lineup optimization (default: 20)")
    parser.add_argument("--size", dest="lineup_size", type=int, default=None, metavar="N",
                        help="Pick exactly N players with no token constraint")
    parser.add_argument("--lineups", dest="lineups_path", default=None, metavar="FILE",
                        help="CSV/text file of preset lineups to evaluate (one per line, comma-separated player names); skips optimization")
    parser.add_argument("--k-factor", dest="k_factor", type=float, default=0, metavar="K",
                        help="Elo K-factor for live updates during simulation (0 = disabled, try 32–64)")
    parser.add_argument("--bo5", dest="bo5", action="store_true",
                        help="Adjust win probabilities for best-of-five matches (default: best-of-three)")
    parser.add_argument("--draw-efficiency", dest="draw_efficiency", action="store_true",
                        help="Print a table of all players sorted by draw efficiency (actual EV / expected EV for their Elo)")
    parser.add_argument("--advancements", dest="advancements_raw", default=None, metavar="PLAYER:ROUND,...",
                        help="Force players to win through a given round, e.g. \"Djokovic:QF,Sinner:F\"")
    parser.add_argument("--boost", dest="boost_raw", default=None, metavar="PLAYER:AMT,...",
                        help="Comma-separated Elo boosts, e.g. \"Sinner:50,Djokovic:-25\"")
    args = parser.parse_args()

    if not args.update_elo and not args.draw_path:
        parser.print_usage()
        raise SystemExit(1)

    if args.update_elo:
        update_elo_files()
    else:
        optimizer = ComprehensiveFantasyOptimizer(
            costs_path=args.costs_path,
            draw_path=args.draw_path,
            elo_path=args.elo_path,
            men_path=args.men_path,
            women_path=args.women_path,
            elo_col=args.elo_col,
            discord=not args.markdown,
            top_n=args.top_n,
            n_simulations=args.n_simulations or (10000 if args.best_at else 0),
            analyze=args.analyze,
            ev_floor=args.ev_floor,
            best_at=args.best_at,
            scoring_rounds=args.scoring_rounds,
            token_cap=args.token_cap,
            lineups_path=args.lineups_path,
            k_factor=args.k_factor,
            lineup_size=args.lineup_size,
            bo5=args.bo5,
            draw_efficiency=args.draw_efficiency,
        )
        optimizer.load_data()

        if args.advancements_raw:
            advancements = {}
            for entry in args.advancements_raw.split(","):
                entry = entry.strip()
                if not entry:
                    continue
                if ":" not in entry:
                    print(f"ERROR: --advancements entry '{entry}' must be in Player:Round format")
                    continue
                raw_name, _, raw_round = entry.rpartition(":")
                try:
                    resolved = _resolve_player(raw_name.strip(), optimizer.players)
                    gender = optimizer.players[resolved]["gender"]
                    max_rounds = optimizer._max_rounds(gender)
                    rnd = optimizer._label_to_round(raw_round.strip(), max_rounds)
                    advancements[resolved] = rnd
                    print(f"Advancing {resolved} through {raw_round.strip().upper()} (round {rnd})", flush=True)
                except ValueError as e:
                    print(f"ERROR: {e}")
            optimizer.advancements = advancements
            optimizer._section_cache.clear()

        if args.boost_raw:
            for entry in args.boost_raw.split(","):
                entry = entry.strip()
                if not entry:
                    continue
                if ":" not in entry:
                    print(f"ERROR: --boost entry '{entry}' must be in Player:amount format")
                    continue
                raw_name, _, raw_amt = entry.rpartition(":")
                try:
                    resolved = _resolve_player(raw_name.strip(), optimizer.players)
                    amt = float(raw_amt.strip())
                    optimizer.players[resolved]["elo"] += amt
                    optimizer._section_cache.clear()
                    print(f"Boosting {resolved} by {amt:+.0f} → {optimizer.players[resolved]['elo']:.0f}", flush=True)
                except ValueError as e:
                    print(f"ERROR: {e}")

        if args.exclude_raw:
            excluded = set()
            for raw in args.exclude_raw.split(","):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    resolved = _resolve_player(raw, optimizer.players)
                    excluded.add(resolved)
                    print(f"Excluding: {resolved}", flush=True)
                except ValueError as e:
                    print(f"ERROR: {e}")
            optimizer.excluded = excluded

        if args.include_raw:
            included = set()
            for raw in args.include_raw.split(","):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    resolved = _resolve_player(raw, optimizer.players)
                    if not optimizer.players[resolved]["is_priced"]:
                        print(f"ERROR: {resolved} is not a priced player and cannot be included")
                        continue
                    included.add(resolved)
                    print(f"Including: {resolved}", flush=True)
                except ValueError as e:
                    print(f"ERROR: {e}")
            optimizer.included = included

        if args.path_player:
            ignored = [f"--{f}" for f, v in [
                ("top", args.top_n != 1),
                ("analyze", args.analyze),
                ("best-at", args.best_at),
                ("ev-floor", args.ev_floor != 0.5),
                ("exclude", bool(args.exclude_raw)),
                ("include", bool(args.include_raw)),
                ("lineups", bool(args.lineups_path)),
            ] if v]
            if ignored:
                print(f"WARNING: --path ignores {', '.join(ignored)}", flush=True)
            try:
                resolved = _resolve_player(args.path_player, optimizer.players)
                optimizer.print_path(resolved)
                n_sims = args.n_simulations or (10000 if args.best_at else 0)
                if n_sims > 0:
                    print()
                    optimizer._print_path_simulations(resolved, n_sims)
            except ValueError as e:
                print(f"ERROR: {e}")
        else:
            optimizer.run_optimization()
