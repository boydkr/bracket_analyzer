# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the tool

```bash
# Basic usage (requires draw CSV, defaults to atp_elo.csv + wta_elo.csv)
python3 bracket_analyzer.py -b draw.csv -p costs.csv

# Surface-specific Elo
python3 bracket_analyzer.py -b draw.csv -p costs.csv -m atp_elo.csv --grass

# Player path analysis
python3 bracket_analyzer.py -b draw.csv -p costs.csv --path "Jannik Sinner"

# Monte Carlo simulation
python3 bracket_analyzer.py -b draw.csv -p costs.csv --simulate 10000 --best-at

# Update Elo ratings from tennisabstract.com
python3 bracket_analyzer.py --update-elo

# Markdown output instead of Discord code blocks
python3 bracket_analyzer.py -b draw.csv -p costs.csv --markdown
```

No build step, no dependencies beyond the Python standard library. Use `python3`.

## Module structure

The codebase is split into focused modules with a strict dependency hierarchy:

```
name_matching.py   (0 deps) — fuzzy name normalization and lookup
elo_math.py        (0 deps) — Elo win probability, BO3/BO5 conversion
formatting.py      (0 deps) — pct(), fmt_player(), fixed_table()
elo_fetcher.py     (0 deps) — scrapes tennisabstract.com
scoring.py         (0 deps) — ScoringModel dataclass
data_loader.py     → name_matching — CSV parsing, BracketData container
bracket.py         → elo_math, data_loader, scoring — tree traversal, EV computation
simulator.py       → elo_math, data_loader, scoring — Monte Carlo simulation
optimizer.py       → data_loader, scoring — branch-and-bound DFS
rows.py            → formatting, optimizer, data_loader — build display rows
output.py          → formatting — render pre-built rows to stdout
bracket_analyzer.py (CLI) → all of the above
```

No circular deps. Each module is independently testable by passing in a `BracketData` object.

## Core data structures

**`BracketData`** (defined in `data_loader.py`) is the shared data container passed to every module:
- `players: dict` — `{name: {gender, cost, line, quadrant, elo, is_priced}}`. BYE sentinels have names like `__BYE_M_3__` and `elo=0.0`.
- `gender_max_rounds: dict` — `{"M": 7, "F": 7}` for a 128-draw; log2 of the draw size.
- `line_index: dict` — `{(gender, line): player_dict}` — reverse lookup by bracket position.
- `line_to_name: dict` — `{(gender, line): name}` — line → canonical name.
- `section_cache: dict` — memoizes `section_win_probs()` results. **Must be cleared** after any mutation to `players[*]["elo"]` or `advancements` (done by `apply_advancements` and `apply_boost` in `bracket_analyzer.py`).

**`ScoringModel`** (defined in `scoring.py`) replaces the old `scoring_rounds: int` parameter everywhere:
- `final_rounds: int` — how many rounds before and including the win award points (default 3 = QF/SF/F/W).
- `points_per_round: int` — points per threshold crossed (default 2).
- `min_round(max_rounds)` — first **1-based** round index that scores. Used for round iteration.
- `prob_start_idx(max_rounds)` — **0-based** index into `all_probs` where scoring begins. Used for array slicing.
- These two differ by 1: `prob_start_idx = min_round - 1`. Getting this wrong causes off-by-one EV errors.

**`Config`** (defined in `bracket_analyzer.py`) holds all CLI options passed to orchestration functions. Fields mirror CLI flags directly.

**`player_evs` dict** — returned by `compute_ev()` and used throughout:
```python
{
    "p_qf": float,      # P(reaching QF)
    "p_sf": float,      # P(reaching SF)
    "p_f":  float,      # P(reaching Final)
    "p_ch": float,      # P(winning championship)
    "ev":   float,      # expected fantasy points
    "all_probs": list,  # [P(1 win), P(2 wins), ..., P(max_rounds wins)]
    "draw_eff": float,  # optional: actual_ev / neutral_ev
    "neutral_ev": float # optional: EV against field-average opponents
}
```

## Data flow

1. `load_data()` (data_loader.py) reads draw + costs + Elo CSVs, does fuzzy name-matching across files, injects BYE sentinels for every empty bracket line, computes `gender_max_rounds`, and returns a `BracketData`.

2. `compute_ev(data, player_name, advancements, bo5, model)` (bracket.py) traverses the bracket tree. For each round, `section_win_probs()` recursively computes the probability that each opponent in that section reaches that round. `expected_win_prob()` sums over those candidates weighted by their survival probability. Results are cached in `data.section_cache`.

3. `find_top_lineups(data, player_evs, n, ...)` (optimizer.py) runs a branch-and-bound DFS over the lineup space, pruning branches whose EV upper bound falls below the current nth-best lineup. Returns `(top_lineups, evaluated, ev_histogram)`.

4. `simulate_tournament(data, gender, rng, ...)` (simulator.py) resolves one full draw by random draws weighted by Elo win probability. `run_simulations()` runs N trials and returns a sorted score list per lineup.

5. Orchestration in `bracket_analyzer.py`: `run_optimization()` computes EVs → prints player pool → finds lineups via `run_standard_optimization()` or `run_best_at_optimization()`. `run_path()` handles single-player path analysis.

## Key design decisions

**BO5 adjustment** (`--bo5`): `elo_math.py` precomputes `_ELO_BO5_LOOKUP` at import time — a 1000-entry table mapping BO3 Elo diff → equivalent BO5 Elo diff. The conversion finds p_per_game via binary search on the BO3 win formula, then computes BO5 win probability and back-converts to Elo diff. Applied only when `bo5=True` and `gender == "M"`.

**`advancements` dict**: `{player_name: round_index}` forces a player to win through a given round. Round index is 1-based (e.g., QF in a 128-draw = round 5). When a player is forced through round r, any match in rounds ≤ r against that player returns `win_prob=0.0` for the other player. The cache must be cleared after setting advancements.

**Branch-and-bound pruning**: The DFS maintains a sorted top-N list. An `ev_suffix` array (precomputed suffix sums of player EVs sorted descending) lets the search bound the maximum EV achievable by any extension of a partial combo. Branches where `forced_ev + ev_upper_bound ≤ nth_best` are skipped. The `objective` callable (default: gross EV sum) must be monotone in individual player EVs for this to be correct.

**`draw_efficiency`**: `compute_draw_efficiency()` computes what a player's EV would be if they faced field-average opponents each round (weighted by survival probability). `draw_efficiency = actual_ev / neutral_ev`. Values > 1.0 mean the draw is favorable. Requires EVs of all other players to already be computed.

**Name matching** (name_matching.py): Three-pass resolution — (1) normalized exact match (strips accents, lowercases, collapses hyphens/apostrophes to spaces), (2) substring match, (3) token overlap fuzzy match. Unmatched players in costs fall back to Elo 1650; a warning is printed. All cross-file lookups use `_resolve_player()` or `_lookup_normalized()`.

**Output modes**: `OutputConfig(discord=True)` (default) wraps tables in triple-backtick blocks for Discord. `--markdown` produces GFM pipe tables. `rows.py` builds data; `output.py` renders it — these are separate so the data layer has no format dependency.

## CSV formats

**Draw** (`-b`): `player,line,gender` — `line` is 1-indexed bracket position (power-of-2 draw size auto-detected from max line). Omit `gender` column for single-gender draws.

**Costs** (`-p`): `player,cost,gender` — integer token cost per player. Omit for no-cost mode (shows player pool only; runs `--best-at` simulation if `--simulate` given).

**Elo** (`-e`/`-m`/`-w`): `player,elo,helo,celo,gelo` — surface columns optional. Use `-m`/`-w` for gender-split files (default: `atp_elo.csv`/`wta_elo.csv`), `-e` for a single file. If the requested surface column is absent, falls back to the first available surface column with a warning.

Example files are in `examples/`.

## Scoring model

Each player earns `points_per_round` (default 2) pts per scoring threshold reached. `--scoring-rounds N` (default 3) means the final N rounds + winning all award points. A champion in a 7-round draw with `--scoring-rounds 3` scores at rounds 5 (QF), 6 (SF), 7 (F), and the win — earning `2 × (3 + 1) = 8` pts. The optimizer maximizes gross portfolio EV subject to the token cap (`--tokens`, default 20).

## Verification commands

After any change, verify against these three baselines:

```bash
# Main lineup optimizer (men + women, grass Elo)
python3 bracket_analyzer.py -b examples/wimbledon_2026_draw.csv \
  -p examples/wimbledon_2026_costs.csv -m atp_elo.csv -w wta_elo.csv --grass

# Single-gender no-costs draw
python3 bracket_analyzer.py -b examples/worldcup2026_bracket.csv \
  -p examples/worldcup_costs.csv -e examples/worldcup2026_elo.csv

# Path analysis
python3 bracket_analyzer.py -b examples/wimbledon_2026_draw.csv \
  -m atp_elo.csv --path "Jannik Sinner" --grass
```
