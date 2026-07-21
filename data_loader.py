import csv
import math
from dataclasses import dataclass, field

from name_matching import (
    _build_norm_index,
    _lookup_normalized,
    _fuzzy_matches,
    _resolve_player,
)


@dataclass
class BracketData:
    players: dict          # {name: {gender, cost, line, quadrant, elo, is_priced}}
    gender_max_rounds: dict  # {gender: int}
    line_index: dict       # {(gender, line): player_dict}
    line_to_name: dict     # {(gender, line): name}
    section_cache: dict = field(default_factory=dict)  # memoization; clear after elo/advancement mutations


def _get_quadrant(line, size):
    q = size // 4
    if line <= q: return 1
    elif line <= q * 2: return 2
    elif line <= q * 3: return 3
    else: return 4


def _load_elo_file(path, elo_col, raw_elos, surface_warned):
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = {k.lower(): k for k in reader.fieldnames}
        surface_cols = [c for c in ("gelo", "celo", "helo") if c in headers]
        if elo_col != "elo" and elo_col not in headers:
            if path not in surface_warned:
                surface_warned.add(path)
                if surface_cols:
                    fallback = surface_cols[0]
                    print(f"WARNING: '{elo_col}' column not found in {path}; "
                          f"using '{fallback}' instead (available: {', '.join(surface_cols)})", flush=True)
                else:
                    print(f"WARNING: '{elo_col}' column not found in {path}; "
                          f"falling back to 'elo'", flush=True)
            col = surface_cols[0] if surface_cols else "elo"
        else:
            if elo_col in headers:
                col = elo_col
            else:
                any_elo = next((c for c in ("elo", "gelo", "celo", "helo") if c in headers), None)
                if any_elo is None:
                    raise ValueError(f"No usable elo column found in {path} (checked: elo, gelo, celo, helo)")
                if path not in surface_warned:
                    surface_warned.add(path)
                    print(f"WARNING: 'elo' column not found in {path}; using '{any_elo}' instead", flush=True)
                col = any_elo
        for row in reader:
            name = row[headers["player"]].strip()
            val = row[headers[col]].strip()
            if val:
                raw_elos[name] = float(val)


def load_data(draw_path, costs_path, elo_path, men_path, women_path, elo_col):
    raw_costs = {}
    raw_draws = {}
    raw_elos = {}

    # Parse Draws (required)
    if not draw_path:
        raise ValueError("A draw CSV is required (-d). No default draw is available.")
    with open(draw_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = {k.lower(): k for k in reader.fieldnames}
        has_gender = "gender" in headers
        for row in reader:
            name = row[headers["player"]].strip()
            line = int(row[headers["line"]])
            gender = row[headers["gender"]].strip().upper() if has_gender else None
            raw_draws[name] = {"line": line, "gender": gender, "quadrant": 0}

    # If gender column was absent or all blank, assign a single default gender
    genders_in_draw = {d["gender"] for d in raw_draws.values() if d["gender"]}
    if not genders_in_draw:
        for d in raw_draws.values():
            d["gender"] = "M"
    elif len(genders_in_draw) == 1:
        single = next(iter(genders_in_draw))
        for d in raw_draws.values():
            if not d["gender"]:
                d["gender"] = single

    # Parse Costs — default to cost=1 for every player in the draw
    if costs_path:
        with open(costs_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = {k.lower(): k for k in reader.fieldnames}
            has_gender = "gender" in headers
            draw_norm_early = _build_norm_index(raw_draws.keys())
            for row in reader:
                name = row[headers["player"]].strip()
                if has_gender:
                    gender = row[headers["gender"]].strip().upper()
                else:
                    draw_match = _lookup_normalized(name, draw_norm_early)
                    gender = raw_draws[draw_match]["gender"] if draw_match else "M"
                raw_costs[name] = {"gender": gender, "cost": int(row[headers["cost"]])}
    else:
        for name, draw_data in raw_draws.items():
            raw_costs[name] = {"gender": draw_data["gender"], "cost": 1}

    # Parse Elos
    surface_warned = set()
    for path in filter(None, [elo_path, men_path, women_path]):
        try:
            _load_elo_file(path, elo_col, raw_elos, surface_warned)
        except FileNotFoundError:
            pass

    # Build normalized indexes for cross-file lookups
    draw_norm = _build_norm_index(raw_draws.keys())
    elo_norm  = _build_norm_index(raw_elos.keys())

    # Require at least one draw player to have an Elo match
    if not any(_lookup_normalized(name, elo_norm) for name in raw_draws):
        raise ValueError(
            "No Elo ratings matched any player in the draw. "
            "Provide an Elo file with -m/-w (gender-split) or -e (gender-neutral), "
            "or check that player names align."
        )

    # Name-matching warnings
    for name in raw_costs:
        draw_match = _lookup_normalized(name, draw_norm)
        if not draw_match:
            continue
        elo_match = _lookup_normalized(name, elo_norm)
        if elo_match is None:
            close = _fuzzy_matches(name, raw_elos.keys())
            hint = f" (did you mean: {', '.join(close)}?)" if close else ""
            print(f"WARNING: '{name}' in costs not found in elo data{hint} — using 1650 fallback", flush=True)
        elif elo_match != name:
            print(f"WARNING: '{name}' in costs matched elo entry '{elo_match}' via normalization", flush=True)

    costs_norm = _build_norm_index(raw_costs.keys())

    # Merge into players dict
    players = {}
    for name, draw_data in raw_draws.items():
        is_priced = name in raw_costs
        if not is_priced:
            costs_match = _lookup_normalized(name, costs_norm)
            if costs_match:
                is_priced = True
                raw_costs[name] = raw_costs[costs_match]
        cost = raw_costs[name]["cost"] if is_priced else 1
        gender = draw_data["gender"]
        elo_key = _lookup_normalized(name, elo_norm)
        elo_val = raw_elos[elo_key] if elo_key else 1650.0
        players[name] = {
            "gender": gender,
            "cost": cost,
            "line": draw_data["line"],
            "quadrant": draw_data["quadrant"],
            "elo": elo_val,
            "is_priced": is_priced,
        }

    # Inject BYE sentinels and compute per-gender draw sizes
    gender_max_rounds = {}
    for gender in ("M", "F"):
        gender_lines = {pd["line"] for pd in players.values() if pd["gender"] == gender}
        if gender_lines:
            max_line = max(gender_lines)
            size = 1
            while size < max_line:
                size *= 2
            gender_max_rounds[gender] = int(math.log2(size))
            for pd in players.values():
                if pd["gender"] == gender:
                    pd["quadrant"] = _get_quadrant(pd["line"], size)
            for ln in range(1, size + 1):
                if ln not in gender_lines:
                    bye_name = f"__BYE_{gender}_{ln}__"
                    players[bye_name] = {
                        "gender": gender,
                        "cost": 0,
                        "line": ln,
                        "quadrant": _get_quadrant(ln, size),
                        "elo": 0.0,
                        "is_priced": False,
                    }

    line_index = {(p["gender"], p["line"]): p for p in players.values()}
    line_to_name = {(pd["gender"], pd["line"]): name for name, pd in players.items()}

    return BracketData(
        players=players,
        gender_max_rounds=gender_max_rounds,
        line_index=line_index,
        line_to_name=line_to_name,
    )


def load_results(path):
    """Read a winner,loser CSV. Returns list of (winner_str, loser_str) raw name pairs."""
    pairs = []
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = {k.lower(): k for k in reader.fieldnames}
        for row in reader:
            winner = row[headers["winner"]].strip()
            loser = row[headers["loser"]].strip()
            if winner and loser:
                pairs.append((winner, loser))
    return pairs


def load_preset_lineups(lineups_path, data, player_evs):
    """Read lineups from file (one per line, comma-separated names).
    Returns list of (ev, tuple_of_names) sorted descending by EV."""
    lineups = []
    errors = []
    with open(lineups_path, mode="r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, 1):
            raw_line = raw_line.strip()
            if not raw_line or raw_line.startswith("#"):
                continue
            parts = [p.strip() for p in raw_line.split(",") if p.strip()]
            resolved = []
            dropped = []
            for name in parts:
                try:
                    resolved.append(_resolve_player(name, data.players))
                except ValueError:
                    dropped.append(name)
            if dropped:
                errors.append(f"  Line {lineno}: dropped {', '.join(dropped)}")
            if not resolved:
                continue
            ev = round(sum(player_evs[p]["ev"] for p in resolved), 3)
            lineups.append((ev, tuple(resolved)))
    if errors:
        print("WARNING: some players not found in draw and were dropped:")
        for e in errors:
            print(e)
    lineups.sort(key=lambda x: -x[0])
    return lineups
