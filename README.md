# bracket_analyzer.py

Fantasy bracket optimizer for tennis (and other sports) tournaments. Given a player pool with token costs, a bracket draw, and Elo ratings, it finds the highest-EV lineup subject to a token cap, runs Monte Carlo simulations, and analyses individual bracket paths.

## Quick start

```bash
./bracket_analyzer.py -b draw.csv -p costs.csv -m atp_elo.csv -w wta_elo.csv
```

## CSV formats

### Costs (`-p`)

One row per player in your fantasy pool.

```
player,cost,gender
Argentina,7,M
Norway,2,M
Spain,7,M
```

| Column | Required | Notes |
|--------|----------|-------|
| `player` | yes | Name must loosely match the draw and Elo files (case-insensitive, accent/hyphen tolerant) |
| `cost` | yes | Integer token cost |
| `gender` | yes | `M` or `F` |

### Draw (`-b`)

Full bracket assignment — one row per team/player.

```
player,line,gender
Argentina,124,M
Spain,4,M
France,8,M
```

| Column | Required | Notes |
|--------|----------|-------|
| `player` | yes | |
| `line` | yes | Integer bracket position (1–128 for a 128-player draw) |
| `gender` | yes | `M` or `F` (omit column if single-gender draw) |

Lines are 1-indexed. Players in the same half of the bracket can only meet in the Final. Unoccupied lines become BYEs automatically.

### Elo (`-e` / `-m` / `-w`)

Ratings file — one row per player. Use `-e` for a single gender-neutral file, or `-m`/`-w` for separate men's/women's files. The default ATP/WTA files (`atp_elo.csv` / `wta_elo.csv`) are fetched from tennisabstract.com via `--update-elo`.

```
player,elo,helo,celo,gelo
Argentina,2144,,,
Jannik Sinner,2319.8,2263.2,2215.7,2088.3
```

| Column | Required | Notes |
|--------|----------|-------|
| `player` | yes | |
| `elo` | yes* | Overall Elo rating |
| `helo` | no | Hard-court Elo (used with `--hard`) |
| `celo` | no | Clay-court Elo (used with `--clay`) |
| `gelo` | no | Grass-court Elo (used with `--grass`) |

\* If `elo` is absent, the tool uses the first available surface column and prints a warning.

Players in the costs/draw files with no Elo match default to **1650**. A warning is printed for each unmatched name.

## Options

```
-b <file>               Draw/bracket CSV
-p <file>               Costs CSV
-e <file>               Gender-neutral Elo CSV
-m <file>               Men's Elo CSV (default: atp_elo.csv)
-w <file>               Women's Elo CSV (default: wta_elo.csv)

--grass / --clay / --hard   Use surface-specific Elo column (warns + auto-falls back if absent)
--scoring-rounds N      Score for reaching the final N rounds + winning (default: 3 = QF/SF/F + W)

--path PLAYER           Show bracket path analysis for one player
--top N                 Show top N lineups by EV (default: 1)
--ev-floor N            Also show all lineups within N EV of optimal (e.g. --ev-floor 1.0)
--simulate N            Run N Monte Carlo simulations and show score percentiles
--lineups <file>        Evaluate preset lineups from a file (one per line, comma-separated names)
--analyze               Show player frequency and pair co-occurrence across top EV lineups
--exclude PLAYERS       Comma-separated players to remove from consideration (fuzzy matched)
--include PLAYERS       Comma-separated players to force into every lineup (fuzzy matched)
--best-at               Show best lineup by P(score ≥ k) at each scoring threshold
--markdown              Output as Markdown tables instead of Discord code blocks
--update-elo            Fetch latest Elo ratings from tennisabstract.com
```

## Scoring

Each player earns **2 points** per scoring threshold they reach. `--scoring-rounds N` (default 3) awards points for reaching each of the final N rounds, plus winning. A champion always scores `2 × (N + 1)` points total.

| `--scoring-rounds` | Scoring thresholds (128-draw) | Max pts/player |
|---|---|---|
| 0 | W only | 2 |
| 1 | F, W | 4 |
| 3 | QF, SF, F, W *(default)* | 8 |
| 4 | R16, QF, SF, F, W | 10 |
| 6 | All rounds | 14 |

## Output columns

| Column | Meaning |
|--------|---------|
| `QF%` / `SF%` / `F%` / `W%` | Probability (%) of reaching that round |
| `EV` | Expected fantasy points |
| `EV/Tok` | EV per token cost — the efficiency metric (top 5 marked with `*`) |
| `StdDev` | Standard deviation of fantasy score |
| `Quad` | Bracket quadrant (Q1–Q4) |

Percentages use up to 3 significant characters: `.04` for < 1%, `8.2` for < 10%, `62` for ≥ 10%.

## Examples

**Optimal lineup:**
```bash
./bracket_analyzer.py -b worldcup_bracket.csv -p worldcup_costs.csv -e worldcup_elo.csv
```

**Player path breakdown with simulation:**
```bash
./bracket_analyzer.py -b draw.csv -e elo.csv --path Argentina --simulate 10000
```

**Top 5 lineups with simulation:**
```bash
./bracket_analyzer.py -b draw.csv -p costs.csv -m atp_elo.csv --top 5 --simulate 10000
```

**Clay-court event, excluding a player:**
```bash
./bracket_analyzer.py -b draw.csv -p costs.csv -m atp_elo.csv --clay --exclude "Novak Djokovic"
```

**Best-at lineup search:**
```bash
./bracket_analyzer.py -b draw.csv -p costs.csv -m atp_elo.csv -w wta_elo.csv --best-at
```

**Evaluate preset lineups:**
```bash
./bracket_analyzer.py -b draw.csv -e elo.csv --lineups my_lineups.txt --simulate 10000
```

**Update default Elo files:**
```bash
./bracket_analyzer.py --update-elo
```
