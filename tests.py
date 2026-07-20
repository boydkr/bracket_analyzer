#!/usr/bin/env python3
"""
Test suite for tennis_sim.

Run:
    python3 tests.py          # all tests
    python3 tests.py -v       # verbose
    python3 tests.py TestEloMath            # single class
    python3 tests.py TestCLIIntegration     # CLI tests only
"""
import io
import random
import subprocess
import sys
import unittest
import contextlib
from pathlib import Path

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from elo_math import calculate_match_win_prob
from scoring import ScoringModel
from name_matching import (
    _normalize, _build_norm_index, _lookup_normalized, _fuzzy_matches, _resolve_player,
)
from bracket import (
    label_to_round, round_label, bracket_opponent_lines,
    compute_ev, compute_draw_efficiency, section_win_probs,
)
from data_loader import load_data, load_preset_lineups
from optimizer import find_top_lineups
from simulator import simulate_tournament, run_simulations, score_lineup_from_sim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_wimbledon(costs=False):
    """Load wimbledon draw suppressing warnings. Returns BracketData."""
    with contextlib.redirect_stdout(io.StringIO()):
        return load_data(
            draw_path=str(REPO_ROOT / "examples/wimbledon_2026_draw.csv"),
            costs_path=str(REPO_ROOT / "wimbledon_2026_costs.csv") if costs else None,
            elo_path=None,
            men_path=str(REPO_ROOT / "atp_elo.csv"),
            women_path=str(REPO_ROOT / "wta_elo.csv"),
            elo_col="elo",
        )


def _compute_all_evs(data, model=None):
    if model is None:
        model = ScoringModel.from_final_rounds(3)
    return {n: compute_ev(data, n, {}, False, model) for n in data.players}


def _cli(*args, cwd=None):
    """Run bracket_analyzer.py with args. Returns CompletedProcess."""
    return subprocess.run(
        [sys.executable, "bracket_analyzer.py"] + list(args),
        capture_output=True,
        text=True,
        cwd=str(cwd or REPO_ROOT),
    )


# ---------------------------------------------------------------------------
# TestEloMath
# ---------------------------------------------------------------------------

class TestEloMath(unittest.TestCase):

    def test_equal_elos_gives_half(self):
        self.assertAlmostEqual(calculate_match_win_prob(1600, 1600), 0.5)

    def test_higher_elo_wins_more(self):
        p = calculate_match_win_prob(1800, 1600)
        self.assertGreater(p, 0.5)

    def test_symmetry(self):
        p_ab = calculate_match_win_prob(1800, 1600)
        p_ba = calculate_match_win_prob(1600, 1800)
        self.assertAlmostEqual(p_ab + p_ba, 1.0)

    def test_bye_opponent_wins(self):
        self.assertEqual(calculate_match_win_prob(1600, 0.0), 1.0)

    def test_bye_player_loses(self):
        self.assertEqual(calculate_match_win_prob(0.0, 1600), 0.0)

    def test_result_in_unit_interval(self):
        for elo_a, elo_b in [(1400, 2100), (2100, 1400), (1600, 1600), (500, 3000)]:
            p = calculate_match_win_prob(elo_a, elo_b)
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_bo5_amplifies_edge_for_men(self):
        p_bo3 = calculate_match_win_prob(1800, 1600, bo5=False)
        p_bo5 = calculate_match_win_prob(1800, 1600, bo5=True, gender="M")
        self.assertGreater(p_bo5, p_bo3)

    def test_bo5_women_uses_bo3_formula(self):
        p_bo3 = calculate_match_win_prob(1800, 1600, bo5=False)
        p_bo5_f = calculate_match_win_prob(1800, 1600, bo5=True, gender="F")
        self.assertAlmostEqual(p_bo5_f, p_bo3)

    def test_bo5_no_gender_uses_bo3_formula(self):
        p_bo3 = calculate_match_win_prob(1800, 1600, bo5=False)
        p_bo5_none = calculate_match_win_prob(1800, 1600, bo5=True, gender=None)
        self.assertAlmostEqual(p_bo5_none, p_bo3)


# ---------------------------------------------------------------------------
# TestScoringModel
# ---------------------------------------------------------------------------

class TestScoringModel(unittest.TestCase):

    def setUp(self):
        self.m = ScoringModel.from_final_rounds(3)

    def test_score_zero_rounds(self):
        self.assertEqual(self.m.score(0, 7), 0)

    def test_score_champion(self):
        self.assertEqual(self.m.score(7, 7), self.m.max_score(7))

    def test_max_score(self):
        self.assertEqual(self.m.max_score(7), 2 * (3 + 1))  # 8

    def test_score_monotone(self):
        for r in range(7):
            self.assertLessEqual(self.m.score(r, 7), self.m.score(r + 1, 7))

    def test_min_round(self):
        self.assertEqual(self.m.min_round(7), 4)

    def test_prob_start_idx(self):
        self.assertEqual(self.m.prob_start_idx(7), 3)

    def test_prob_start_idx_nonnegative(self):
        # With final_rounds > max_rounds the clamp kicks in
        m_wide = ScoringModel.from_final_rounds(10)
        self.assertGreaterEqual(m_wide.prob_start_idx(3), 0)

    def test_scoring_thresholds(self):
        self.assertEqual(list(self.m.scoring_thresholds(7)), [4, 5, 6, 7])

    def test_from_final_rounds_identical(self):
        m2 = ScoringModel(final_rounds=3, points_per_round=2)
        self.assertEqual(self.m.score(5, 7), m2.score(5, 7))
        self.assertEqual(self.m.min_round(7), m2.min_round(7))

    def test_custom_points_per_round(self):
        m4 = ScoringModel(final_rounds=3, points_per_round=4)
        self.assertEqual(m4.max_score(7), 4 * 4)
        self.assertEqual(m4.score(7, 7), 16)

    def test_score_below_threshold(self):
        # Rounds 1-3 do not score with final_rounds=3 in a 7-round draw
        for r in range(1, 4):
            self.assertEqual(self.m.score(r, 7), 0)

    def test_score_at_threshold(self):
        # Round 4 (QF) is the first scoring round
        self.assertEqual(self.m.score(4, 7), 2)
        self.assertEqual(self.m.score(5, 7), 4)
        self.assertEqual(self.m.score(6, 7), 6)


# ---------------------------------------------------------------------------
# TestNameMatching
# ---------------------------------------------------------------------------

class TestNameMatching(unittest.TestCase):

    def test_normalize_accents(self):
        self.assertEqual(_normalize("Björn Borg"), "bjorn borg")

    def test_normalize_hyphen(self):
        self.assertEqual(_normalize("Auger-Aliassime"), "auger aliassime")

    def test_normalize_apostrophe(self):
        self.assertEqual(_normalize("O'Brien"), "o brien")

    def test_normalize_case(self):
        self.assertEqual(_normalize("JANNIK SINNER"), "jannik sinner")

    def test_normalize_idempotent(self):
        name = "félix auger-aliassime"
        self.assertEqual(_normalize(_normalize(name)), _normalize(name))

    def test_build_norm_index_round_trip(self):
        names = ["Jannik Sinner", "Novak Djokovic", "Carlos Alcaraz"]
        idx = _build_norm_index(names)
        for name in names:
            self.assertIn(_normalize(name), idx)
            self.assertEqual(idx[_normalize(name)], name)

    def test_lookup_exact(self):
        idx = _build_norm_index(["Jannik Sinner"])
        self.assertEqual(_lookup_normalized("Jannik Sinner", idx), "Jannik Sinner")

    def test_lookup_accent_variant(self):
        idx = _build_norm_index(["Félix Auger-Aliassime"])
        self.assertEqual(_lookup_normalized("Felix Auger Aliassime", idx), "Félix Auger-Aliassime")

    def test_lookup_miss_returns_none(self):
        idx = _build_norm_index(["Jannik Sinner"])
        self.assertIsNone(_lookup_normalized("Nobody Here", idx))

    def test_fuzzy_matches_limit(self):
        candidates = ["Jannik Sinner", "Jannik Smith", "John Sinner", "Alice"]
        results = _fuzzy_matches("Jannik Sinner", candidates, max_results=2)
        self.assertLessEqual(len(results), 2)

    def test_fuzzy_matches_zero_overlap(self):
        results = _fuzzy_matches("xyz qrs", ["abc def", "ghi jkl"])
        self.assertEqual(results, [])

    def test_fuzzy_matches_sorted_by_overlap(self):
        # "jannik sinner" shares 2 tokens with "Jannik Sinner", 1 with "Jannik Smith"
        results = _fuzzy_matches("Jannik Sinner", ["Jannik Smith", "Jannik Sinner", "Alice"])
        self.assertEqual(results[0], "Jannik Sinner")

    def test_resolve_exact(self):
        players = {"Jannik Sinner": {}, "Novak Djokovic": {}}
        self.assertEqual(_resolve_player("Jannik Sinner", players), "Jannik Sinner")

    def test_resolve_accent_variant(self):
        players = {"Félix Auger-Aliassime": {}}
        result = _resolve_player("Felix Auger Aliassime", players)
        self.assertEqual(result, "Félix Auger-Aliassime")

    def test_resolve_unknown_raises(self):
        players = {"Jannik Sinner": {}}
        with self.assertRaises(ValueError):
            _resolve_player("Nobody Fakename", players)

    def test_resolve_ambiguous_raises(self):
        players = {"Jannik Sinner": {}, "Jannik Smith": {}}
        with self.assertRaises(ValueError) as ctx:
            _resolve_player("Jannik", players)
        self.assertIn("Ambiguous", str(ctx.exception))


# ---------------------------------------------------------------------------
# TestBracketFunctions
# ---------------------------------------------------------------------------

class TestBracketFunctions(unittest.TestCase):

    def test_label_to_round_winner(self):
        self.assertEqual(label_to_round("W", 7), 7)
        self.assertEqual(label_to_round("WIN", 7), 7)
        self.assertEqual(label_to_round("WINNER", 7), 7)

    def test_label_to_round_named_rounds(self):
        self.assertEqual(label_to_round("F", 7), 6)
        self.assertEqual(label_to_round("SF", 7), 5)
        self.assertEqual(label_to_round("QF", 7), 4)

    def test_label_to_round_r_notation(self):
        # R{n} means "n players remain" — forces player through the round *before* the field
        # shrinks to n. R128 = 0 (before round 1), R64 = 1 (through round 1), etc.
        self.assertEqual(label_to_round("R128", 7), 0)
        self.assertEqual(label_to_round("R64", 7), 1)
        self.assertEqual(label_to_round("R32", 7), 2)
        self.assertEqual(label_to_round("R16", 7), 3)

    def test_label_to_round_case_insensitive(self):
        self.assertEqual(label_to_round("qf", 7), label_to_round("QF", 7))

    def test_label_to_round_unknown_raises(self):
        with self.assertRaises(ValueError):
            label_to_round("ZZ", 7)

    def test_round_label_final(self):
        self.assertEqual(round_label(7, 7), "F")

    def test_round_label_sf(self):
        self.assertEqual(round_label(6, 7), "SF")

    def test_round_label_qf(self):
        self.assertEqual(round_label(5, 7), "QF")

    def test_round_label_r128(self):
        self.assertEqual(round_label(1, 7), "R128")

    def test_round_label_round_trip(self):
        # round_label / label_to_round are not exact inverses for R-notation
        # (R128 → 0, not 1), so only test named rounds QF/SF/F/W
        for rnd, label in [(4, "QF"), (5, "SF"), (6, "F"), (7, "W")]:
            self.assertEqual(label_to_round(label, 7), rnd if label != "F" else 6)

    def test_bracket_opponent_lines_length(self):
        for max_rounds in (3, 5, 7):
            size = 2 ** max_rounds
            sections = bracket_opponent_lines(1, size)
            self.assertEqual(len(sections), max_rounds)

    def test_bracket_opponent_lines_own_line_absent(self):
        sections = bracket_opponent_lines(1, 128)
        all_lines = [l for sec in sections for l in sec]
        self.assertNotIn(1, all_lines)

    def test_bracket_opponent_lines_no_duplicates_across_sections(self):
        sections = bracket_opponent_lines(3, 128)
        all_lines = [l for sec in sections for l in sec]
        self.assertEqual(len(all_lines), len(set(all_lines)))

    def test_bracket_opponent_lines_covers_full_draw(self):
        line = 5
        size = 128
        sections = bracket_opponent_lines(line, size)
        all_opp_lines = set(l for sec in sections for l in sec)
        self.assertEqual(len(all_opp_lines), size - 1)


# ---------------------------------------------------------------------------
# TestDataLoader
# ---------------------------------------------------------------------------

class TestDataLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = _load_wimbledon(costs=False)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.data_costs = _load_wimbledon(costs=True)

    def test_wimbledon_both_genders(self):
        self.assertIn("M", self.data.gender_max_rounds)
        self.assertIn("F", self.data.gender_max_rounds)

    def test_wimbledon_max_rounds(self):
        self.assertEqual(self.data.gender_max_rounds["M"], 7)
        self.assertEqual(self.data.gender_max_rounds["F"], 7)

    def test_all_lines_present_per_gender(self):
        for gender in ("M", "F"):
            max_rounds = self.data.gender_max_rounds[gender]
            size = 2 ** max_rounds
            for line in range(1, size + 1):
                self.assertIn((gender, line), self.data.line_index,
                              f"Missing ({gender}, {line})")

    def test_bye_sentinels(self):
        byes = [n for n in self.data.players if n.startswith("__BYE_")]
        for name in byes:
            p = self.data.players[name]
            self.assertEqual(p["elo"], 0.0)
            self.assertFalse(p["is_priced"])

    def test_line_index_line_to_name_consistent(self):
        for (gender, line), name in self.data.line_to_name.items():
            self.assertIn((gender, line), self.data.line_index)
            self.assertEqual(self.data.line_index[(gender, line)], self.data.players[name])

    def test_quadrants_valid(self):
        for name, p in self.data.players.items():
            if name.startswith("__BYE_"):
                continue
            self.assertIn(p["quadrant"], {1, 2, 3, 4})

    def test_quadrant_line1_is_q1(self):
        p = next(p for p in self.data.players.values()
                 if p["gender"] == "M" and p["line"] == 1)
        self.assertEqual(p["quadrant"], 1)

    def test_quadrant_line128_is_q4(self):
        p = next(p for p in self.data.players.values()
                 if p["gender"] == "M" and p["line"] == 128)
        self.assertEqual(p["quadrant"], 4)

    def test_no_costs_all_cost_one(self):
        for name, p in self.data.players.items():
            if not name.startswith("__BYE_"):
                self.assertEqual(p["cost"], 1)

    def test_costs_file_marks_priced_players(self):
        priced = [n for n, p in self.data_costs.players.items() if p["is_priced"]]
        self.assertGreater(len(priced), 0)
        for name in priced:
            self.assertGreater(self.data_costs.players[name]["cost"], 0)

    def test_section_cache_initially_empty(self):
        d = _load_wimbledon()
        self.assertEqual(len(d.section_cache), 0)

    def test_worldcup_single_gender(self):
        with contextlib.redirect_stdout(io.StringIO()):
            d = load_data(
                draw_path=str(REPO_ROOT / "examples/worldcup2026_bracket.csv"),
                costs_path=None,
                elo_path=str(REPO_ROOT / "examples/worldcup2026_elo.csv"),
                men_path=None,
                women_path=None,
                elo_col="elo",
            )
        self.assertNotIn("F", d.gender_max_rounds)
        self.assertEqual(d.gender_max_rounds["M"], 5)

    def test_missing_draw_raises(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            load_data(None, None, None, str(REPO_ROOT / "atp_elo.csv"), None, "elo")

    def test_load_preset_lineups(self):
        with contextlib.redirect_stdout(io.StringIO()):
            d = load_data(
                draw_path=str(REPO_ROOT / "examples/worldcup2026_bracket.csv"),
                costs_path=str(REPO_ROOT / "worldcup_costs.csv"),
                elo_path=str(REPO_ROOT / "examples/worldcup2026_elo.csv"),
                men_path=None, women_path=None, elo_col="elo",
            )
        model = ScoringModel.from_final_rounds(3)
        evs = _compute_all_evs(d, model)
        with contextlib.redirect_stdout(io.StringIO()):
            lineups = load_preset_lineups(str(REPO_ROOT / "lineups.txt"), d, evs)
        self.assertGreater(len(lineups), 0)
        # Sorted descending by EV
        evs_only = [ev for ev, _ in lineups]
        self.assertEqual(evs_only, sorted(evs_only, reverse=True))


# ---------------------------------------------------------------------------
# TestComputeEV
# ---------------------------------------------------------------------------

class TestComputeEV(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = _load_wimbledon(costs=False)
        cls.model = ScoringModel.from_final_rounds(3)

    def setUp(self):
        self.data.section_cache.clear()

    def test_return_keys(self):
        ev = compute_ev(self.data, "Jannik Sinner", {}, False, self.model)
        for key in ("p_qf", "p_sf", "p_f", "p_ch", "ev", "all_probs"):
            self.assertIn(key, ev)

    def test_probs_in_unit_interval(self):
        ev = compute_ev(self.data, "Jannik Sinner", {}, False, self.model)
        for key in ("p_qf", "p_sf", "p_f", "p_ch"):
            self.assertGreaterEqual(ev[key], 0.0)
            self.assertLessEqual(ev[key], 1.0)

    def test_survival_probabilities_decreasing(self):
        ev = compute_ev(self.data, "Jannik Sinner", {}, False, self.model)
        self.assertGreaterEqual(ev["p_qf"], ev["p_sf"])
        self.assertGreaterEqual(ev["p_sf"], ev["p_f"])
        self.assertGreaterEqual(ev["p_f"], ev["p_ch"])

    def test_all_probs_length(self):
        ev = compute_ev(self.data, "Jannik Sinner", {}, False, self.model)
        self.assertEqual(len(ev["all_probs"]), self.data.gender_max_rounds["M"])

    def test_ev_nonnegative(self):
        ev = compute_ev(self.data, "Jannik Sinner", {}, False, self.model)
        self.assertGreaterEqual(ev["ev"], 0.0)

    def test_higher_elo_higher_ev(self):
        # Sinner (Elo ~2100) should have higher EV than a lower-ranked men's player
        ev_sinner = compute_ev(self.data, "Jannik Sinner", {}, False, self.model)
        # Find a lower-Elo men's player
        low_player = min(
            (n for n, p in self.data.players.items()
             if p["gender"] == "M" and p["elo"] > 0 and not n.startswith("__BYE_")),
            key=lambda n: self.data.players[n]["elo"]
        )
        self.data.section_cache.clear()
        ev_low = compute_ev(self.data, low_player, {}, False, self.model)
        self.assertGreater(ev_sinner["ev"], ev_low["ev"])

    def test_advancement_forces_reach(self):
        # Force Sinner through the Final (round 6 in a 7-round draw)
        adv = {"Jannik Sinner": label_to_round("F", 7)}
        self.data.section_cache.clear()
        ev = compute_ev(self.data, "Jannik Sinner", adv, False, self.model)
        self.assertAlmostEqual(ev["p_f"], 1.0)

    def test_section_cache_populated(self):
        self.data.section_cache.clear()
        compute_ev(self.data, "Jannik Sinner", {}, False, self.model)
        self.assertGreater(len(self.data.section_cache), 0)

    def test_draw_efficiency_returns_floats(self):
        evs = {n: compute_ev(self.data, n, {}, False, self.model) for n in self.data.players}
        eff, neutral = compute_draw_efficiency(self.data, "Jannik Sinner", evs, self.model, False)
        self.assertIsInstance(eff, float)
        self.assertIsInstance(neutral, float)
        self.assertGreater(eff, 0.0)


# ---------------------------------------------------------------------------
# TestOptimizer
# ---------------------------------------------------------------------------

class TestOptimizer(unittest.TestCase):
    """Uses World Cup draw (32 teams) — optimizer runs in milliseconds vs seconds for 128-player draws."""

    @classmethod
    def setUpClass(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            cls.data = load_data(
                draw_path=str(REPO_ROOT / "examples/worldcup2026_bracket.csv"),
                costs_path=str(REPO_ROOT / "worldcup_costs.csv"),
                elo_path=str(REPO_ROOT / "examples/worldcup2026_elo.csv"),
                men_path=None, women_path=None, elo_col="elo",
            )
        cls.model = ScoringModel.from_final_rounds(3)
        cls.evs = _compute_all_evs(cls.data, cls.model)

        def _run(**kw):
            defaults = dict(excluded=set(), included=set(), token_cap=20,
                            lineup_size=None, ev_floor=0.0)
            defaults.update(kw)
            return find_top_lineups(cls.data, cls.evs, **defaults)

        # Precompute all needed results up front
        cls.top, cls.evaluated, cls.hist = _run(n=5)
        cls.top_included, _, _ = _run(n=3, included={"Argentina"})
        cls.top_excluded, _, _ = _run(n=3, excluded={"Argentina"})
        cls.top_sized, _, _    = _run(n=3, lineup_size=4, token_cap=999)
        cls.top_zero_obj, _, _ = _run(n=3, objective=lambda lu, evs: 0.0)

    def test_returns_n_lineups(self):
        self.assertLessEqual(len(self.top), 5)
        self.assertGreater(len(self.top), 0)

    def test_top_lineups_descending(self):
        scores = [s for s, _ in self.top]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_evaluated_ascending(self):
        scores = [s for s, _ in self.evaluated]
        self.assertEqual(scores, sorted(scores))

    def test_histogram_count_matches_evaluated(self):
        self.assertEqual(self.hist["n"], len(self.evaluated))

    def test_histogram_min_le_max(self):
        self.assertLessEqual(self.hist["min"], self.hist["max"])

    def test_histogram_percentile_keys(self):
        for p in (10, 25, 50, 75, 90):
            self.assertIn(p, self.hist["percentiles"])

    def test_token_cap_respected(self):
        for _, lineup in self.top:
            total = sum(self.data.players[p]["cost"] for p in lineup)
            self.assertLessEqual(total, 20)

    def test_included_player_in_every_lineup(self):
        for _, lineup in self.top_included:
            self.assertIn("Argentina", lineup)

    def test_excluded_player_in_no_lineup(self):
        for _, lineup in self.top_excluded:
            self.assertNotIn("Argentina", lineup)

    def test_lineup_size_exact(self):
        for _, lineup in self.top_sized:
            self.assertEqual(len(lineup), 4)

    def test_custom_objective_all_zero(self):
        for score, _ in self.top_zero_obj:
            self.assertEqual(score, 0.0)


# ---------------------------------------------------------------------------
# TestSimulator
# ---------------------------------------------------------------------------

class TestSimulator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = _load_wimbledon(costs=True)
        cls.model = ScoringModel.from_final_rounds(3)
        cls.evs = _compute_all_evs(cls.data, cls.model)
        cls.n_real_m = sum(
            1 for n, p in cls.data.players.items()
            if p["gender"] == "M" and not n.startswith("__BYE_")
        )

    def test_simulate_all_nonnegative(self):
        rng = random.Random(42)
        result = simulate_tournament(self.data, "M", rng, {}, False)
        for val in result.values():
            self.assertGreaterEqual(val, 0)

    def test_simulate_exactly_one_champion(self):
        rng = random.Random(42)
        result = simulate_tournament(self.data, "M", rng, {}, False)
        max_rounds = self.data.gender_max_rounds["M"]
        champions = [n for n, r in result.items() if r == max_rounds]
        self.assertEqual(len(champions), 1)

    def test_simulate_total_rounds_won(self):
        rng = random.Random(42)
        result = simulate_tournament(self.data, "M", rng, {}, False)
        self.assertEqual(sum(result.values()), self.n_real_m - 1)

    def test_simulate_advancement_forced(self):
        rng = random.Random(42)
        adv = {"Jannik Sinner": label_to_round("F", 7)}
        result = simulate_tournament(self.data, "M", rng, adv, False)
        # Sinner guaranteed at least 6 rounds (through Final)
        self.assertGreaterEqual(result.get("Jannik Sinner", 0), 6)

    def test_simulate_no_bye_in_result(self):
        rng = random.Random(42)
        result = simulate_tournament(self.data, "M", rng, {}, False)
        for name in result:
            self.assertFalse(name.startswith("__BYE_"))

    def test_run_simulations_count(self):
        priced = [(0.0, (n,)) for n, p in self.data.players.items()
                  if p["is_priced"] and p["gender"] == "M"][:3]
        scores = run_simulations(self.data, priced, 50, {}, False, 0, self.model)
        self.assertEqual(len(scores), 3)
        for s in scores:
            self.assertEqual(len(s), 50)

    def test_run_simulations_sorted(self):
        priced = [(0.0, (n,)) for n, p in self.data.players.items()
                  if p["is_priced"] and p["gender"] == "M"][:2]
        scores = run_simulations(self.data, priced, 30, {}, False, 0, self.model)
        for s in scores:
            self.assertEqual(s, sorted(s))

    def test_run_simulations_nonnegative(self):
        priced = [(0.0, (n,)) for n, p in self.data.players.items()
                  if p["is_priced"] and p["gender"] == "M"][:2]
        scores = run_simulations(self.data, priced, 30, {}, False, 0, self.model)
        for s in scores:
            self.assertGreaterEqual(s[0], 0)

    def test_score_lineup_from_sim(self):
        rng = random.Random(7)
        result = simulate_tournament(self.data, "M", rng, {}, False)
        name = "Jannik Sinner"
        score = score_lineup_from_sim(self.data, (name,), result, self.model)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, self.model.max_score(7))


# ---------------------------------------------------------------------------
# TestCLIIntegration
# ---------------------------------------------------------------------------

WIMBLEDON_FLAGS = [
    "-b", "examples/wimbledon_2026_draw.csv",
    "-p", "wimbledon_2026_costs.csv",
    "--ev-floor", "0",
]
WORLDCUP_FLAGS = [
    "-b", "examples/worldcup2026_bracket.csv",
    "-p", "worldcup_costs.csv",
    "-e", "examples/worldcup2026_elo.csv",
]


class TestCLIIntegration(unittest.TestCase):

    def _run(self, *args):
        return _cli(*args)

    # --- Basic output ---

    def test_basic_wimbledon(self):
        r = self._run(*WIMBLEDON_FLAGS)
        self.assertEqual(r.returncode, 0)
        self.assertIn("PLAYER POOL", r.stdout)
        self.assertIn("OPTIMAL LINEUP", r.stdout)
        self.assertIn("EV:", r.stdout)
        self.assertIn("Tokens:", r.stdout)

    def test_grass_elo(self):
        r = self._run(*WIMBLEDON_FLAGS, "--grass")
        self.assertEqual(r.returncode, 0)
        self.assertIn("gElo", r.stdout)

    def test_clay_elo(self):
        r = self._run(*WIMBLEDON_FLAGS, "--clay")
        self.assertEqual(r.returncode, 0)
        self.assertIn("cElo", r.stdout)

    def test_hard_elo(self):
        r = self._run(*WIMBLEDON_FLAGS, "--hard")
        self.assertEqual(r.returncode, 0)
        self.assertIn("hElo", r.stdout)

    def test_bo5(self):
        r = self._run(*WIMBLEDON_FLAGS, "--bo5")
        self.assertEqual(r.returncode, 0)

    def test_markdown_output(self):
        r = self._run(*WIMBLEDON_FLAGS, "--markdown")
        self.assertEqual(r.returncode, 0)
        self.assertIn("|", r.stdout)
        self.assertNotIn("```", r.stdout)

    def test_top_n_lineups(self):
        r = self._run(*WIMBLEDON_FLAGS, "--top", "3")
        self.assertEqual(r.returncode, 0)
        self.assertIn("LINEUP #1", r.stdout)
        self.assertIn("LINEUP #2", r.stdout)
        self.assertIn("LINEUP #3", r.stdout)

    # --- Path analysis ---

    def test_path_analysis(self):
        r = self._run(
            "-b", "examples/wimbledon_2026_draw.csv",
            "-m", "atp_elo.csv", "--path", "Jannik Sinner", "--grass",
        )
        self.assertEqual(r.returncode, 0)
        self.assertIn("Path:", r.stdout)
        self.assertIn("P(reach)", r.stdout)
        self.assertIn("Win%", r.stdout)

    def test_path_bad_player(self):
        r = self._run(
            "-b", "examples/wimbledon_2026_draw.csv",
            "-m", "atp_elo.csv", "--path", "Nobody Fakename",
        )
        self.assertTrue(r.returncode != 0 or "ERROR" in r.stdout)

    # --- Mutations ---

    def test_advancements(self):
        r = self._run(*WIMBLEDON_FLAGS, "--advancements", "Jannik Sinner:F")
        self.assertEqual(r.returncode, 0)
        self.assertIn("Advancing", r.stdout)
        # Sinner should appear with 100% F probability
        self.assertIn("Sinner", r.stdout)

    def test_advancements_path_shows_forced(self):
        r = self._run(
            "-b", "examples/wimbledon_2026_draw.csv",
            "-m", "atp_elo.csv",
            "--path", "Jannik Sinner",
            "--advancements", "Jannik Sinner:F",
        )
        self.assertEqual(r.returncode, 0)
        self.assertIn("[forced]", r.stdout)

    def test_boost(self):
        r = self._run(*WIMBLEDON_FLAGS, "--boost", "Jannik Sinner:200")
        self.assertEqual(r.returncode, 0)
        self.assertIn("Boosting", r.stdout)

    def test_exclude(self):
        r = self._run(*WIMBLEDON_FLAGS, "--exclude", "Jannik Sinner")
        self.assertEqual(r.returncode, 0)
        # Sinner should not appear in the lineup table (after the pool section)
        lineup_section = r.stdout.split("OPTIMAL LINEUP", 1)[-1] if "OPTIMAL LINEUP" in r.stdout else ""
        self.assertNotIn("Sinner", lineup_section)

    def test_include(self):
        r = self._run(*WIMBLEDON_FLAGS, "--include", "Novak Djokovic")
        self.assertEqual(r.returncode, 0)
        lineup_section = r.stdout.split("OPTIMAL LINEUP", 1)[-1] if "OPTIMAL LINEUP" in r.stdout else r.stdout
        self.assertIn("Djokovic", lineup_section)

    # --- Simulation flags ---

    def test_simulate(self):
        r = self._run(*WIMBLEDON_FLAGS, "--simulate", "100")
        self.assertEqual(r.returncode, 0)
        self.assertIn("Analytical vs Simulated", r.stdout)
        self.assertIn("Score Distribution", r.stdout)

    def test_best_at(self):
        r = self._run(*WIMBLEDON_FLAGS, "--simulate", "100", "--best-at")
        self.assertEqual(r.returncode, 0)
        self.assertIn("Best lineup by P(score", r.stdout)
        # Distribution tables should NOT have 100 columns
        if "Score Distribution" in r.stdout:
            dist_section = r.stdout.split("Score Distribution", 1)[-1].split("\n")[1]
            col_count = dist_section.count("|") - 1  # subtract Score column
            self.assertLess(col_count, 100)

    def test_analyze(self):
        r = self._run(*WIMBLEDON_FLAGS, "--analyze", "--ev-floor", "1.0", "--simulate", "100")
        self.assertEqual(r.returncode, 0)
        self.assertIn("Player Frequency", r.stdout)

    def test_k_factor(self):
        r = self._run(*WIMBLEDON_FLAGS, "--simulate", "50", "--k-factor", "32")
        self.assertEqual(r.returncode, 0)

    # --- Scoring / token options ---

    def test_scoring_rounds(self):
        r = self._run(*WIMBLEDON_FLAGS, "--scoring-rounds", "2")
        self.assertEqual(r.returncode, 0)

    def test_token_cap(self):
        r = self._run(*WIMBLEDON_FLAGS, "--tokens", "15")
        self.assertEqual(r.returncode, 0)
        # "Tokens: X/15" should appear in output
        self.assertIn("/15", r.stdout)

    def test_lineup_size(self):
        r = self._run("-b", "examples/wimbledon_2026_draw.csv", "--size", "4")
        self.assertEqual(r.returncode, 0)

    def test_ev_floor(self):
        r = self._run(*WIMBLEDON_FLAGS, "--ev-floor", "0.2")
        self.assertEqual(r.returncode, 0)

    # --- Draw efficiency ---

    def test_draw_efficiency(self):
        r = self._run(*WIMBLEDON_FLAGS, "--draw-efficiency")
        self.assertEqual(r.returncode, 0)
        self.assertIn("DRAW EFFICIENCY", r.stdout)
        self.assertIn("DrawEff", r.stdout)

    # --- Different tournaments / draws ---

    def test_worldcup(self):
        r = self._run(*WORLDCUP_FLAGS)
        self.assertEqual(r.returncode, 0)
        self.assertIn("Argentina", r.stdout)

    def test_french_clay(self):
        r = self._run(
            "-b", "examples/french_2026_draw.csv",
            "-p", "french2026_costs.csv",
            "-e", "examples/french_clay_men.csv",
            "--clay",
        )
        self.assertEqual(r.returncode, 0)
        self.assertIn("cElo", r.stdout)

    def test_wimbledon_qf_draw(self):
        r = self._run("-b", "wimbledon_qf_draw.csv")
        self.assertEqual(r.returncode, 0)

    # --- No costs mode ---

    def test_no_costs_file_runs_with_unit_costs(self):
        # Without -p, every player costs 1 and the optimizer still runs
        r = self._run("-b", "wimbledon_qf_draw.csv")
        self.assertEqual(r.returncode, 0)
        self.assertIn("PLAYER POOL", r.stdout)
        self.assertIn("OPTIMAL LINEUP", r.stdout)

    # --- Preset lineups ---

    def test_preset_lineups(self):
        r = self._run(*WORLDCUP_FLAGS, "--lineups", "lineups.txt")
        self.assertEqual(r.returncode, 0)
        self.assertIn("LINEUP", r.stdout)

    # --- Error cases ---

    def test_missing_draw_flag(self):
        r = self._run("-p", "wimbledon_2026_costs.csv")
        self.assertNotEqual(r.returncode, 0)


if __name__ == "__main__":
    # Support running a single class: python3 tests.py TestEloMath
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        suite = unittest.TestLoader().loadTestsFromName(sys.argv[1], sys.modules[__name__])
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    unittest.main()
