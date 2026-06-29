#!/usr/bin/env python3
import csv
import itertools
import math
import unicodedata


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


def _normalize(name):
    """Lowercase, strip accents, collapse punctuation to spaces."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
    result = []
    for ch in ascii_name.lower():
        if ch.isalpha() or ch.isspace():
            result.append(ch)
        elif ch in "-'":
            result.append(" ")
    return " ".join("".join(result).split())


def _build_norm_index(names):
    """Return {normalized_name: canonical_name} for a collection of names."""
    return {_normalize(n): n for n in names}


def _lookup_normalized(name, norm_index):
    """Return canonical name from norm_index if found, else None."""
    return norm_index.get(_normalize(name))


def _fuzzy_matches(name, candidates, max_results=3):
    """Return up to max_results candidates whose normalized tokens overlap with name."""
    name_tokens = set(_normalize(name).split())
    scored = []
    for c in candidates:
        c_tokens = set(_normalize(c).split())
        overlap = len(name_tokens & c_tokens)
        if overlap:
            scored.append((overlap, c))
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:max_results]]


def _resolve_player(name, players):
    """Return exact player name from players dict, or raise with helpful message."""
    if name in players:
        return name
    # Try normalized match (handles case, hyphens, accents)
    norm_index = _build_norm_index(players.keys())
    match = _lookup_normalized(name, norm_index)
    if match:
        return match
    # Try substring on normalized forms
    name_norm = _normalize(name)
    matches = [p for p in players if name_norm in _normalize(p) or _normalize(p) in name_norm]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise ValueError(f"Ambiguous player '{name}'. Matches: {', '.join(sorted(matches))}")
    close = _fuzzy_matches(name, players.keys())
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise ValueError(f"Player '{name}' not found in draw.{hint}")


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
        self.lineups_path = lineups_path
        self.k_factor = k_factor
        self.lineup_size = lineup_size
        self.bo5 = bo5
        self._gender_max_rounds = {}
        self.players = {}

    def _get_quadrant(self, line, size=128):
        q = size // 4
        if line <= q: return 1
        elif line <= q * 2: return 2
        elif line <= q * 3: return 3
        else: return 4

    def _round_label(self, rnd, max_rounds):
        """Label for the rnd-th match played in a max_rounds-round draw.
        rnd=1 is the first match (R128 in a 128-draw), rnd=max_rounds is the Final."""
        rounds_remaining_before = max_rounds - rnd + 1  # rounds left including this one
        if rounds_remaining_before == 1: return "F"
        if rounds_remaining_before == 2: return "SF"
        if rounds_remaining_before == 3: return "QF"
        return f"R{2**rounds_remaining_before}"

    def _max_rounds(self, gender):
        return self._gender_max_rounds.get(gender, 7)

    def load_data(self):
        raw_costs = {}
        raw_draws = {}
        raw_elos = {}

        # Parse Draws (required)
        if not self.draw_path:
            raise ValueError("A draw CSV is required (-d). No default draw is available.")
        with open(self.draw_path, mode="r", encoding="utf-8") as f:
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
        if self.costs_path:
            with open(self.costs_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                headers = {k.lower(): k for k in reader.fieldnames}
                has_gender = "gender" in headers
                draw_norm_early = _build_norm_index(raw_draws.keys())
                for row in reader:
                    name = row[headers["player"]].strip()
                    if has_gender:
                        gender = row[headers["gender"]].strip().upper()
                    else:
                        # inherit gender from draw
                        draw_match = _lookup_normalized(name, draw_norm_early)
                        gender = raw_draws[draw_match]["gender"] if draw_match else "M"
                    raw_costs[name] = {"gender": gender, "cost": int(row[headers["cost"]])}
        else:
            for name, draw_data in raw_draws.items():
                raw_costs[name] = {"gender": draw_data["gender"], "cost": 1}

        # Parse Elos — gender-neutral (-e), men (-m), women (-w)
        _surface_warned = set()
        def load_elo_file(path):
            with open(path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                headers = {k.lower(): k for k in reader.fieldnames}
                surface_cols = [c for c in ("gelo", "celo", "helo") if c in headers]
                if self.elo_col != "elo" and self.elo_col not in headers:
                    if path not in _surface_warned:
                        _surface_warned.add(path)
                        if surface_cols:
                            fallback = surface_cols[0]
                            print(f"WARNING: '{self.elo_col}' column not found in {path}; "
                                  f"using '{fallback}' instead (available: {', '.join(surface_cols)})", flush=True)
                        else:
                            print(f"WARNING: '{self.elo_col}' column not found in {path}; "
                                  f"falling back to 'elo'", flush=True)
                    col = surface_cols[0] if surface_cols else "elo"
                else:
                    if self.elo_col in headers:
                        col = self.elo_col
                    else:
                        any_elo = next((c for c in ("elo", "gelo", "celo", "helo") if c in headers), None)
                        if any_elo is None:
                            raise ValueError(f"No usable elo column found in {path} (checked: elo, gelo, celo, helo)")
                        if path not in _surface_warned:
                            _surface_warned.add(path)
                            print(f"WARNING: 'elo' column not found in {path}; using '{any_elo}' instead", flush=True)
                        col = any_elo
                for row in reader:
                    name = row[headers["player"]].strip()
                    val = row[headers[col]].strip()
                    if val:
                        raw_elos[name] = float(val)

        for path in filter(None, [self.elo_path, self.men_path, self.women_path]):
            try:
                load_elo_file(path)
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

        # Name-matching warnings: costs players missing from elo (only if they're in the draw)
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

        # Merge fields tracking unpriced status
        for name, draw_data in raw_draws.items():
            is_priced = name in raw_costs
            if not is_priced:
                # Check if a costs entry normalizes to this draw name
                costs_match = _lookup_normalized(name, costs_norm)
                if costs_match:
                    is_priced = True
                    raw_costs[name] = raw_costs[costs_match]
            cost = raw_costs[name]["cost"] if is_priced else 1
            gender = draw_data["gender"]
            # Use normalized elo lookup so hyphen/accent variants resolve correctly
            elo_key = _lookup_normalized(name, elo_norm)
            elo_val = raw_elos[elo_key] if elo_key else 1650.0

            self.players[name] = {
                "gender": gender,
                "cost": cost,
                "line": draw_data["line"],
                "quadrant": draw_data["quadrant"],
                "elo": elo_val,
                "is_priced": is_priced
            }

        # Inject BYE sentinels for empty lines within each gender's draw range
        for gender in ("M", "F"):
            gender_lines = {pd["line"] for pd in self.players.values() if pd["gender"] == gender}
            if gender_lines:
                max_line = max(gender_lines)
                size = 1
                while size < max_line:
                    size *= 2
                self._gender_max_rounds[gender] = int(math.log2(size))
                # Fix quadrants for all real players of this gender now that size is known
                for pd in self.players.values():
                    if pd["gender"] == gender:
                        pd["quadrant"] = self._get_quadrant(pd["line"], size)
                full_range = range(1, size + 1)
                for ln in full_range:
                    if ln not in gender_lines:
                        bye_name = f"__BYE_{gender}_{ln}__"
                        self.players[bye_name] = {
                            "gender": gender,
                            "cost": 0,
                            "line": ln,
                            "quadrant": self._get_quadrant(ln, size),
                            "elo": 0.0,
                            "is_priced": False,
                        }

        # Build (gender, line) → player index for bracket lookups
        self._line_index = {
            (p["gender"], p["line"]): p
            for p in self.players.values()
        }
        self._section_cache = {}

    def calculate_match_win_prob(self, elo_a, elo_b):
        if elo_b == 0.0:
            return 1.0
        if elo_a == 0.0:
            return 0.0
        if self.bo5:
            raw_diff = elo_a - elo_b
            abs_diff = min(abs(round(raw_diff)), _MAX_ELO_DIFF)
            adjusted_diff = _ELO_BO5_LOOKUP[abs_diff]
            if raw_diff < 0:
                adjusted_diff = -adjusted_diff
            return 1.0 / (1.0 + 10 ** (-adjusted_diff / 400.0))
        return 1 / (1 + math.pow(10, (elo_b - elo_a) / 400))

    def _bracket_opponent_lines(self, line, size=128):
        """Return max_rounds opponent sections for a draw of the given size.
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

    def _section_win_probs(self, lines, gender):
        """Return {line: P(that player wins the section)} for all known players
        in `lines`, using recursive bracket simulation.  The section must be a
        power-of-two-sized contiguous block; unknown lines are ignored."""
        key = (gender, lines[0], lines[-1])
        if key in self._section_cache:
            return self._section_cache[key]
        known = [l for l in lines if (gender, l) in self._line_index]
        if not known:
            self._section_cache[key] = {}
            return {}
        if len(lines) == 1:
            result = {lines[0]: 1.0} if known else {}
            self._section_cache[key] = result
            return result
        if len(lines) == 2:
            a, b = lines[0], lines[1]
            a_known = (gender, a) in self._line_index
            b_known = (gender, b) in self._line_index
            if a_known and b_known:
                elo_a = self._line_index[(gender, a)]["elo"]
                elo_b = self._line_index[(gender, b)]["elo"]
                p = self.calculate_match_win_prob(elo_a, elo_b)
                result = {a: p, b: 1 - p}
            elif a_known:
                result = {a: 1.0}
            else:
                result = {b: 1.0}
            self._section_cache[key] = result
            return result

        mid = len(lines) // 2
        left, right = lines[:mid], lines[mid:]
        left_probs  = self._section_win_probs(left, gender)
        right_probs = self._section_win_probs(right, gender)

        result = {}
        for l, p_l in left_probs.items():
            elo_l = self._line_index[(gender, l)]["elo"]
            for r, p_r in right_probs.items():
                elo_r = self._line_index[(gender, r)]["elo"]
                p_lr = self.calculate_match_win_prob(elo_l, elo_r)
                result[l] = result.get(l, 0) + p_l * p_r * p_lr
                result[r] = result.get(r, 0) + p_l * p_r * (1 - p_lr)
        self._section_cache[key] = result
        return result

    def _expected_win_prob(self, player_elo, lines, gender, fallback_elo):
        """Σ P(j wins section) × P(player beats j).
        Correct form: averages win probabilities over the opponent distribution
        rather than plugging E[elo] into the nonlinear sigmoid."""
        probs = self._section_win_probs(lines, gender)
        if not probs:
            return self.calculate_match_win_prob(player_elo, fallback_elo)
        return sum(
            p_j * self.calculate_match_win_prob(player_elo, self._line_index[(gender, j)]["elo"])
            for j, p_j in probs.items()
        )

    def compute_ev(self, player_name):
        """Simulate a player's path through the bracket using actual draw opponents
        where available, falling back to generic tier Elos otherwise."""
        p_data = self.players[player_name]
        gender = p_data["gender"]
        p_elo = p_data["elo"]
        line = p_data["line"]
        quad = p_data["quadrant"]
        max_rounds = self._max_rounds(gender)
        size = 2 ** max_rounds

        # Generic fallbacks interpolated from early-round to late-round Elo
        fb_starts = {1: 1500.0, 2: 1520.0, 3: 1510.0, 4: 1530.0}
        fb_end = 1950.0
        start = fb_starts[quad]
        if max_rounds == 1:
            fb = [fb_end]
        else:
            fb = [start + (fb_end - start) * i / (max_rounds - 1) for i in range(max_rounds)]

        opp_sections = self._bracket_opponent_lines(line, size)
        ewp = self._expected_win_prob
        p_reach = 1.0
        all_p = []
        for i, opp_lines in enumerate(opp_sections):
            p_reach = p_reach * ewp(p_elo, opp_lines, gender, fb[i])
            all_p.append(round(p_reach, 4))

        min_idx = max(0, max_rounds - self.scoring_rounds - 1)
        ev = round(2 * sum(all_p[min_idx:]), 3)
        return {
            "p_qf": all_p[-4] if len(all_p) >= 4 else all_p[0],
            "p_sf": all_p[-3] if len(all_p) >= 3 else all_p[0],
            "p_f":  all_p[-2] if len(all_p) >= 2 else all_p[0],
            "p_ch": all_p[-1],
            "ev":        ev,
            "all_probs": all_p,
        }

    def _meeting_block_size(self, name_a, name_b):
        """Smallest power-of-2 block containing both players' lines (same gender only)."""
        la = self.players[name_a]["line"]
        lb = self.players[name_b]["line"]
        gender = self.players[name_a]["gender"]
        size = 2 ** self._max_rounds(gender)
        mbs = 2
        while mbs <= size:
            if (la - 1) // mbs == (lb - 1) // mbs:
                return mbs
            mbs *= 2
        return size

    def _pairwise_cov(self, name_a, name_b, evs):
        """Cov[s_A, s_B] for two players.  Zero for cross-gender (independent draws)."""
        if self.players[name_a]["gender"] != self.players[name_b]["gender"]:
            return 0.0
        mbs = self._meeting_block_size(name_a, name_b)
        gender = self.players[name_a]["gender"]
        max_rounds = self._max_rounds(gender)
        min_idx = max(0, max_rounds - self.scoring_rounds - 1)
        # meeting_idx: 0-based index in all_probs where these two could first meet
        meeting_idx = int(math.log2(mbs)) - 1
        first = max(min_idx, meeting_idx)
        ea = evs[name_a]["all_probs"]
        eb = evs[name_b]["all_probs"]
        return 4.0 * sum(-ea[i] * eb[i] for i in range(first, max_rounds))

    def _score_variance(self, name, evs):
        """Var[s] for a single player's fantasy score."""
        p = evs[name]
        gender = self.players[name]["gender"]
        min_idx = max(0, self._max_rounds(gender) - self.scoring_rounds - 1)
        scoring_probs = p["all_probs"][min_idx:]
        e_s2 = sum((8*(j+1) - 4) * scoring_probs[j] for j in range(len(scoring_probs)))
        return e_s2 - p["ev"] ** 2

    def _lineup_variance(self, lineup, evs):
        """Portfolio variance of the total lineup score."""
        var = sum(self._score_variance(p, evs) for p in lineup)
        for a, b in itertools.combinations(lineup, 2):
            var += 2 * self._pairwise_cov(a, b, evs)
        return var


    @staticmethod
    def _pct(v):
        """Format a probability (0–1) as a percentage string.
        <1%: 2 decimal places, no leading zero (.04); <10%: 1 decimal (8.2); else integer (62)."""
        p = v * 100
        if p < 1:
            return f"{p:.2f}".lstrip("0")
        elif p < 10:
            return f"{p:.1f}"
        else:
            return f"{p:.0f}"

    @staticmethod
    def _fmt(pd, s):
        """Return consistently formatted display values for a player."""
        _pct = ComprehensiveFantasyOptimizer._pct
        return {
            "elo":      str(round(pd["elo"])),
            "p_qf":     _pct(s['p_qf']),
            "p_sf":     _pct(s['p_sf']),
            "p_f":      _pct(s['p_f']),
            "p_ch":     _pct(s['p_ch']),
            "ev":       f"{s['ev']:.2f}",
            "ev_tok":   f"{s['ev']/pd['cost']:.2f}",
        }

    @staticmethod
    def _fixed_table(headers, rows):
        """Return a list of lines for a fixed-width plain-text table."""
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))
        sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
        def fmt_row(cells):
            return "|" + "|".join(f" {str(c):<{widths[i]}} " for i, c in enumerate(cells)) + "|"
        lines = [sep, fmt_row(headers), sep]
        for row in rows:
            lines.append(fmt_row(row))
        lines.append(sep)
        return lines

    def _print_lineup(self, title, note, player_evs, best_lineup):
        elo_label = {"elo": "Elo", "gelo": "gElo", "celo": "cElo", "helo": "hElo"}.get(self.elo_col, self.elo_col)
        gross_ev = round(sum(player_evs[p]["ev"] for p in best_lineup), 3)
        tokens = sum(self.players[p]["cost"] for p in best_lineup)
        portfolio_var = self._lineup_variance(best_lineup, player_evs)
        portfolio_std = math.sqrt(max(portfolio_var, 0))

        if self.discord:
            rows = []
            for p in best_lineup:
                pd = self.players[p]
                s = player_evs[p]
                f = self._fmt(pd, s)
                indiv_std = math.sqrt(max(self._score_variance(p, player_evs), 0))
                rows.append([p, pd["gender"], pd["cost"], f["elo"],
                              f["p_qf"], f["p_sf"], f["p_f"], f["p_ch"],
                              f["ev"], f["ev_tok"], f"{indiv_std:.2f}", f"Q{pd['quadrant']}"])
            headers = ["Player", "G", "Cost", elo_label,
                       "QF%", "SF%", "F%", "W%", "EV", "EV/Tok", "StdDev", "Quad"]
            lines = self._fixed_table(headers, rows)
            print(f"**{title}**")
            if note:
                print(f"_{note}_")
            print(f"EV: {gross_ev:.2f}  |  StdDev: {portfolio_std:.2f}  |  Tokens: {tokens}/{self.token_cap}")
            print("```")
            print("\n".join(lines))
            print("```")
        else:
            print(f"**Total Portfolio EV:** {gross_ev:.2f} Points")
            print(f"**Portfolio StdDev:** {portfolio_std:.2f} Points")
            print(f"**Total Capital Spent:** {tokens} / {self.token_cap} Tokens\n")
            print(f"| Selected Athlete | Gender | Cost | {elo_label} | QF% | SF% | F% | W% | EV | EV/Token | StdDev | Bracket Quadrant |")
            print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
            for p in best_lineup:
                pd = self.players[p]
                s = player_evs[p]
                f = self._fmt(pd, s)
                indiv_std = math.sqrt(max(self._score_variance(p, player_evs), 0))
                print(f"| **{p}** | {pd['gender']} | {pd['cost']} | {f['elo']} "
                      f"| {f['p_qf']} | {f['p_sf']} | {f['p_f']} | {f['p_ch']} "
                      f"| {f['ev']} | {f['ev_tok']} | {indiv_std:.2f} | Quarter {pd['quadrant']} |")

    def _simulate_tournament(self, gender, rng, live_elos=None):
        """Simulate one full gender draw. Returns {player_name: rounds_won} for all players.
        live_elos, if provided, is a mutable {name: elo} dict updated each match (K-factor mode)."""
        survivors = {
            pd["line"]: name
            for name, pd in self.players.items()
            if pd["gender"] == gender
        }
        rounds_won = {
            name: 0 for name, pd in self.players.items()
            if pd["gender"] == gender and not name.startswith("__BYE_")
        }

        n_players = len(survivors)
        while n_players > 1:
            next_survivors = {}
            lines = sorted(survivors)
            for i in range(0, len(lines), 2):
                la, lb = lines[i], lines[i + 1]
                na, nb = survivors[la], survivors[lb]
                elo_a = live_elos[na] if live_elos else self.players[na]["elo"]
                elo_b = live_elos[nb] if live_elos else self.players[nb]["elo"]
                if elo_a == 0.0:
                    next_survivors[lb] = nb
                    if nb in rounds_won:
                        rounds_won[nb] += 1
                elif elo_b == 0.0:
                    next_survivors[la] = na
                    if na in rounds_won:
                        rounds_won[na] += 1
                else:
                    p = self.calculate_match_win_prob(elo_a, elo_b)
                    if rng.random() < p:
                        winner_line, winner, loser, p_win = la, na, nb, p
                    else:
                        winner_line, winner, loser, p_win = lb, nb, na, 1 - p
                    rounds_won[winner] += 1
                    next_survivors[winner_line] = winner
                    if live_elos is not None:
                        k = self.k_factor
                        live_elos[winner] += k * (1 - p_win)
                        live_elos[loser]  += k * (0 - p_win)
            survivors = next_survivors
            n_players = len(survivors)

        return rounds_won

    def _score_lineup_from_sim(self, lineup, rounds_won):
        """Score a lineup against one simulated tournament result.
        2 pts each for reaching the final scoring_rounds rounds."""
        score = 0
        for name in lineup:
            r = rounds_won.get(name, 0)
            gender = self.players[name]["gender"]
            max_rounds = self._max_rounds(gender)
            min_r = max_rounds - self.scoring_rounds
            for threshold in range(min_r, max_rounds + 1):
                if r >= threshold:
                    score += 2
        return score

    _SIM_CALL_CAP = 10_000_000

    def _cap_sim_pool(self, pool, n_trials, label=""):
        cap = max(1, self._SIM_CALL_CAP // n_trials)
        if len(pool) > cap:
            print(
                f"WARNING: {label}{len(pool):,} lineups × {n_trials:,} trials = "
                f"{len(pool)*n_trials:,} calls — capping to top {cap:,} by EV "
                f"(>{len(pool)-cap:,} dropped).",
                flush=True,
            )
            pool = pool[-cap:]  # pool is sorted ascending, tail = highest EV
        return pool

    def run_simulations(self, lineups, n_trials=10000):
        """Run n_trials full-draw simulations. Returns {lineup_index: sorted score list}."""
        import random
        rng = random.Random()
        scores = [[] for _ in lineups]
        use_live_elos = self.k_factor != 0
        base_elos = {name: pd["elo"] for name, pd in self.players.items()} if use_live_elos else None

        for _ in range(n_trials):
            if use_live_elos:
                live_elos = dict(base_elos)
                m_result = self._simulate_tournament("M", rng, live_elos)
                f_result = self._simulate_tournament("F", rng, live_elos)
            else:
                m_result = self._simulate_tournament("M", rng)
                f_result = self._simulate_tournament("F", rng)
            combined = {**m_result, **f_result}
            for i, (_, lineup) in enumerate(lineups):
                scores[i].append(self._score_lineup_from_sim(lineup, combined))

        for s in scores:
            s.sort()
        return scores

    def _find_top_lineups(self, player_evs, n):
        """Return the top-n distinct lineups by gross EV using branch-and-bound DFS."""
        candidates = sorted(
            [p for p in self.players if self.players[p]["is_priced"] and p not in self.excluded],
            key=lambda x: player_evs[x]["ev"], reverse=True,
        )
        # Forced inclusions: validate and pre-deduct their cost/slots
        forced = [p for p in self.included if p in {c for c in candidates}]
        forced_cost = sum(self.players[p]["cost"] for p in forced)
        forced_ev   = sum(player_evs[p]["ev"]    for p in forced)
        forced_slots = len(forced)
        candidates = [p for p in candidates if p not in self.included]
        nc = len(candidates)
        costs_arr = [self.players[p]["cost"] for p in candidates]
        evs_arr   = [player_evs[p]["ev"]    for p in candidates]

        # ev_suffix[i] = sum(evs_arr[i:])  — used for EV upper-bound pruning
        ev_suffix = [0.0] * (nc + 1)
        for i in range(nc - 1, -1, -1):
            ev_suffix[i] = ev_suffix[i + 1] + evs_arr[i]

        if self.lineup_size is not None:
            # Fixed size: exactly lineup_size players, no token constraint
            min_size = max(0, self.lineup_size - forced_slots)
            max_size = max(0, self.lineup_size - forced_slots)
        else:
            # Token-capped: any size from 1 up to all candidates
            min_size = 1
            max_size = nc + forced_slots

        def run_search(prune_floor, collect_all=False):
            results = []
            evaluated = []

            def nth_best():
                return results[-1][0] if len(results) == n else -1.0

            def cutoff():
                # When collecting all lineups in the floor window, hold pruning fixed
                # at prune_floor so nth_best() can't tighten it as results accumulate.
                return prune_floor if collect_all else max(nth_best(), prune_floor)

            def search(idx, combo, cost, ev):
                size = len(combo)

                if size >= min_size:
                    full_combo = tuple(forced) + tuple(combo)
                    total_ev = forced_ev + ev
                    evaluated.append((total_ev, full_combo))
                    cur_nth = nth_best()
                    if total_ev > cur_nth:
                        if len(results) < n:
                            results.append((total_ev, full_combo))
                            if len(results) == n:
                                results.sort(key=lambda x: -x[0])
                        else:
                            results[-1] = (total_ev, full_combo)
                            results.sort(key=lambda x: -x[0])

                if size == max_size:
                    return

                remaining = nc - idx
                min_more  = max(0, min_size - size)
                max_slots = max_size - size

                if remaining < min_more:
                    return

                ev_ub = ev + ev_suffix[idx] - ev_suffix[min(idx + max_slots, nc)]
                if forced_ev + ev_ub <= cutoff():
                    return

                for i in range(idx, nc):
                    if nc - i - 1 < max(0, min_size - size - 1):
                        break

                    new_cost = cost + costs_arr[i]
                    if self.lineup_size is None and new_cost > self.token_cap:
                        continue

                    slots_after = max_slots - 1
                    ev_ub_i = ev + evs_arr[i] + ev_suffix[i + 1] - ev_suffix[min(i + 1 + slots_after, nc)]
                    if forced_ev + ev_ub_i <= cutoff():
                        continue

                    combo.append(candidates[i])
                    search(i + 1, combo, new_cost, ev + evs_arr[i])
                    combo.pop()

            search(0, [], forced_cost, 0.0)
            results.sort(key=lambda x: -x[0])
            return results, evaluated

        # Pass 1: find optimal EV
        results, evaluated = run_search(-1.0)

        # Pass 2: if ev_floor set, re-run keeping all lineups within floor of optimal
        if self.ev_floor > 0 and results:
            optimal_ev = results[0][0]
            floor = optimal_ev - self.ev_floor
            _, evaluated = run_search(floor, collect_all=True)
            results = [(ev, combo) for ev, combo in evaluated if ev >= floor]
            results.sort(key=lambda x: -x[0])
        results.sort(key=lambda x: -x[0])
        evaluated.sort(key=lambda x: x[0])

        evs_only = [e for e, _ in evaluated]
        ne = len(evs_only)
        buckets = {}
        for ev in evs_only:
            b = round(ev * 2) / 2
            buckets[b] = buckets.get(b, 0) + 1
        pct = [10, 25, 50, 75, 90]
        pct_vals = {p: evs_only[min(int(p / 100 * ne), ne - 1)] for p in pct}
        print(f"[optimizer] {ne:,} lineups evaluated  "
              f"min={evs_only[0]:.2f}  "
              + "  ".join(f"P{p}={pct_vals[p]:.2f}" for p in pct)
              + f"  max={evs_only[-1]:.2f}", flush=True)
        print("[optimizer] EV distribution:")
        max_count = max(buckets.values())
        for b in sorted(buckets):
            bar = "█" * int(buckets[b] / max_count * 40)
            print(f"  {b:5.1f}  {bar}  {buckets[b]}")

        return results[:n], evaluated

    def _print_sim_comparison(self, lineups, sim_scores, player_evs, labels=None):
        """Compare analytical EV/stddev vs simulated mean/stddev for each lineup."""
        rows = []
        for i, (ev, lineup) in enumerate(lineups):
            label = labels[i] if labels else f"#{i+1}"
            exp_ev = ev
            exp_std = math.sqrt(max(self._lineup_variance(lineup, player_evs), 0))

            scores = sim_scores[i]
            n = len(scores)
            sim_mean = sum(scores) / n
            sim_std = math.sqrt(sum((s - sim_mean) ** 2 for s in scores) / n)

            rows.append([
                label,
                f"{exp_ev:.2f}", f"{exp_std:.2f}",
                f"{sim_mean:.2f}", f"{sim_std:.2f}",
                f"{sim_mean - exp_ev:+.2f}",
            ])

        headers = ["Lineup", "E[EV]", "E[σ]", "Sim μ", "Sim σ", "Δμ"]
        if self.discord:
            print("**Analytical vs Simulated**")
            print("```")
            print("\n".join(self._fixed_table(headers, rows)))
            print("```")
        else:
            print("**Analytical vs Simulated**\n")
            print("| " + " | ".join(headers) + " |")
            print("| " + " | ".join(["---"] * len(headers)) + " |")
            for row in rows:
                print("| " + " | ".join(row) + " |")
            print()

    def _print_score_distributions(self, lineups, sim_scores, labels=None):
        """Print a side-by-side P(score=k) and P(score>=k) table for all lineups."""
        n = len(sim_scores[0])
        # Determine score range across all lineups
        max_score = max(max(s) for s in sim_scores)
        scores_range = range(0, max_score + 2, 2)

        # Build frequency dicts
        freq = []
        for scores in sim_scores:
            fd = {}
            for s in scores:
                fd[s] = fd.get(s, 0) + 1
            freq.append(fd)

        lineup_labels = labels if labels else [f"#{i+1}" for i in range(len(lineups))]

        # P(score=k) table — exclude rows where all columns are < 0.1%
        p_rows = []
        for k in scores_range:
            vals = [fd.get(k, 0) / n * 100 for fd in freq]
            if max(vals) < 0.1:
                continue
            p_rows.append([str(k)] + [f"{v:.1f}%" for v in vals])

        # P(score>=k) table — precompute all values, then filter rows where
        # no column changed by >= 0.1pp from the previous included row
        ge_all = []
        for k in scores_range:
            if k == 0:
                continue
            vals = [sum(fd.get(s, 0) for s in scores_range if s >= k) / n * 100 for fd in freq]
            ge_all.append((k, vals))

        ge_rows = []
        prev_vals = None
        for k, vals in ge_all:
            if all(v == 0.0 for v in vals):
                continue
            if prev_vals is None or any(abs(v - pv) >= 0.1 for v, pv in zip(vals, prev_vals)):
                ge_rows.append([str(k)] + [f"{v:.1f}%" for v in vals])
                prev_vals = vals

        headers = ["Score"] + lineup_labels

        if self.discord:
            print("**Score Distribution  P(score = k)**")
            print("```")
            print("\n".join(self._fixed_table(headers, p_rows)))
            print("```")
            print("**Exceedance  P(score ≥ k)**")
            print("```")
            print("\n".join(self._fixed_table(headers, ge_rows)))
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

    def _print_analysis(self, evaluated, top_k=100):
        """Player frequency and pairwise co-occurrence across the top-k lineups by EV."""
        if not evaluated:
            return
        members = evaluated[-top_k:]  # evaluated is sorted ascending, so tail = highest EV
        nm = len(members)
        evs = [ev for ev, _ in members]
        min_ev = evs[0]
        max_ev = evs[-1]
        avg_ev = sum(evs) / nm
        std_ev = math.sqrt(sum((e - avg_ev) ** 2 for e in evs) / nm)
        print(f"\n**Analysis: top {nm} lineups  (EV {min_ev:.2f} – {max_ev:.2f},  avg {avg_ev:.2f},  σ {std_ev:.2f})**")

        all_players = sorted({p for _, combo in members for p in combo})

        # Player frequency
        freq = {p: sum(1 for _, combo in members if p in combo) for p in all_players}
        freq_sorted = sorted(freq.items(), key=lambda x: -x[1])
        freq_rows = [[p, str(f), f"{f/nm*100:.0f}%"] for p, f in freq_sorted if f / nm >= 0.20]

        # Individual frequencies as fractions for lift calculation
        freq_frac = {p: f / nm for p, f in freq_sorted}

        # Pairwise lift: observed co-occurrence / (P(A) * P(B))
        # Only for players 20%-95% frequent; filter pairs at >=10% co-occurrence
        common = [p for p, f in freq_sorted]
        pair_rows = []
        for i, pa in enumerate(common):
            for pb in common[i+1:]:
                both = sum(1 for _, combo in members if pa in combo and pb in combo)
                p_both = both / nm
                if both >= 2:
                    expected = freq_frac[pa] * freq_frac[pb]
                    lift = p_both / expected if expected > 0 else 0.0
                    pair_rows.append([f"{pa} + {pb}", str(both), f"{lift:.2f}"])
        pair_rows.sort(key=lambda r: -float(r[2]))
        pair_rows = pair_rows[:20]

        if self.discord:
            print("**Player Frequency**")
            print("```")
            print("\n".join(self._fixed_table(["Player", "Count", "Freq%"], freq_rows)))
            print("```")
            if pair_rows:
                print("**Pairs (lift = observed / expected)**")
                print("```")
                print("\n".join(self._fixed_table(["Pair", "Count", "Lift"], pair_rows)))
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

    def _print_best_player_at(self, player_evs):
        """Simulate individual players, then show best pick by P(score >= k) per threshold."""
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

        max_score = self.scoring_rounds * 2
        THRESHOLDS = [k for k in [2, 4, 6, 8, 10, 12, 14] if k <= max_score]
        rows = []
        for k in THRESHOLDS:
            ge = {name: sum(1 for s in scores[name] if s >= k) / n_trials * 100 for name in priced}
            best = max(priced, key=lambda n: ge[n])
            ev = player_evs[best]["ev"]
            cost = self.players[best]["cost"]
            rows.append([f"≥{k}", f"{ge[best]:.1f}%", f"{ev:.2f}", f"{cost}", best])

        headers = ["Score", "P(≥k)", "EV", "Cost", "Player"]
        if self.discord:
            print(f"\n**Best single pick by P(score ≥ k)**")
            print("```")
            print("\n".join(self._fixed_table(headers, rows)))
            print("```")
        else:
            print(f"\n### Best single pick by P(score ≥ k)\n")
            print("| " + " | ".join(headers) + " |")
            print("| " + " | ".join(["---"] * len(headers)) + " |")
            for row in rows:
                print("| " + " | ".join(row) + " |")
            print()

    def _print_path_simulations(self, player_name, n_trials):
        """Simulate the tournament n_trials times and show:
        1. Simulated vs analytical round-reach rates for this player.
        2. Score distribution P(score=k) and P(score>=k)."""
        import random
        rng = random.Random()
        pd = self.players[player_name]
        gender = pd["gender"]
        max_rounds = self._max_rounds(gender)

        rounds_reached = [0] * (max_rounds + 1)  # index = round number (1-based)
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
        all_p = ev["all_probs"]

        # Generate correct round labels for this draw size
        # Label for winning match rnd = the round just played
        # Detect bye rounds: opponent section contains only BYE sentinels
        line = pd["line"]
        size = 2 ** max_rounds
        opp_sections = self._bracket_opponent_lines(line, size)
        def is_bye_section(opp_lines):
            probs = self._section_win_probs(opp_lines, gender)
            real = [j for j in probs if self._line_index[(gender, j)]["elo"] > 0.0]
            return len(real) == 0

        # Table 1: simulated vs analytical reach rates (skip bye rounds)
        # P(reach round rnd) = P(won rnd-1 matches).  rounds_reached[0] = n_trials (always in draw).
        rounds_reached[0] = n_trials
        rows1 = []
        for rnd in range(1, max_rounds + 1):
            opp_lines = opp_sections[rnd - 1]
            if is_bye_section(opp_lines):
                continue
            label = self._round_label(rnd, max_rounds)
            sim_pct = rounds_reached[rnd - 1] / n_trials
            ana_pct = all_p[rnd - 2] if rnd > 1 else 1.0
            rows1.append([label, self._pct(sim_pct) + "%", self._pct(ana_pct) + "%",
                          f"{(sim_pct - ana_pct)*100:+.1f}pp"])
        # Champion row
        sim_w = rounds_reached[max_rounds] / n_trials
        ana_w = all_p[max_rounds - 1]
        rows1.append(["W", self._pct(sim_w) + "%", self._pct(ana_w) + "%",
                      f"{(sim_w - ana_w)*100:+.1f}pp"])

        headers1 = ["Round", "Sim", "Analytical", "Δ"]

        # Table 2: score distribution
        max_score = max(score_counts.keys()) if score_counts else 0
        scores_range = range(0, max_score + 2, 2)
        p_rows = [[str(k), f"{score_counts.get(k,0)/n_trials*100:.1f}%"]
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

        headers2 = ["Score", "P(=k)"]
        headers3 = ["Score", "P(≥k)"]

        if self.discord:
            print("\n**Simulated vs Analytical reach rates**")
            print("```")
            print("\n".join(self._fixed_table(headers1, rows1)))
            print("```")
            print("**Score Distribution  P(score = k)**")
            print("```")
            print("\n".join(self._fixed_table(headers2, p_rows)))
            print("```")
            print("**Exceedance  P(score ≥ k)**")
            print("```")
            print("\n".join(self._fixed_table(headers3, ge_rows)))
            print("```")
        else:
            print("\n### Simulated vs Analytical reach rates\n")
            print("| " + " | ".join(headers1) + " |")
            print("| " + " | ".join(["---"] * len(headers1)) + " |")
            for row in rows1:
                print("| " + " | ".join(row) + " |")
            print()
            print("### Score Distribution  P(score = k)\n")
            print("| " + " | ".join(headers2) + " |")
            print("| --- | --- |")
            for row in p_rows:
                print("| " + " | ".join(row) + " |")
            print()
            print("### Exceedance  P(score ≥ k)\n")
            print("| " + " | ".join(headers3) + " |")
            print("| --- | --- |")
            for row in ge_rows:
                print("| " + " | ".join(row) + " |")
            print()

    def load_preset_lineups(self, player_evs):
        """Read lineups from self.lineups_path (one lineup per line, comma-separated names).
        Returns list of (ev, tuple_of_names) sorted descending by EV, same shape as _find_top_lineups."""
        lineups = []
        errors = []
        with open(self.lineups_path, mode="r", encoding="utf-8") as f:
            for lineno, raw_line in enumerate(f, 1):
                raw_line = raw_line.strip()
                if not raw_line or raw_line.startswith("#"):
                    continue
                parts = [p.strip() for p in raw_line.split(",") if p.strip()]
                resolved = []
                dropped = []
                for name in parts:
                    try:
                        resolved.append(_resolve_player(name, self.players))
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

            BEST_AT_THRESHOLDS = [2, 4, 6, 8, 10, 12, 14, 16, 18]
            n_trials = len(pool_scores[0])

            # Find winner index per threshold
            threshold_winners = {}
            for k in BEST_AT_THRESHOLDS:
                ge_vals = [sum(1 for s in scores if s >= k) / n_trials * 100 for scores in pool_scores]
                threshold_winners[k] = max(range(len(ge_vals)), key=lambda i: ge_vals[i])

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

            header_line = f"**Best lineup by P(score ≥ k)  (across top-{len(pool)} lineups)**"
            if self.discord:
                print(header_line)
                print("```")
                print("\n".join(self._fixed_table(["Score", "P(≥k)", "EV", "Lineup"], rows)))
                print("```")
            else:
                print(f"{header_line}\n")
                print("| Score | P(≥k) | EV | Lineup |")
                print("| --- | --- | --- | --- |")
                for row in rows:
                    print("| " + " | ".join(row) + " |")
                print()

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
        names = sorted(
            [n for n, p in self.players.items() if p["is_priced"] and p["gender"] == gender],
            key=lambda n: player_evs[n]["ev"],
            reverse=True,
        )
        if top_evtok_names is None:
            top_evtok_names = set()
        if self.discord:
            rows = []
            for name in names:
                pd = self.players[name]
                s = player_evs[name]
                f = self._fmt(pd, s)
                evtok = f["ev_tok"] + "*" if name in top_evtok_names else f["ev_tok"]
                rows.append([name, pd["cost"], f["elo"],
                              f["p_qf"], f["p_sf"], f["p_f"], f["p_ch"],
                              f["ev"], evtok])
            headers = ["Player", "Cost", elo_label, "QF%", "SF%", "F%", "W%", "EV", "EV/Tok"]
            lines = self._fixed_table(headers, rows)
            if title:
                print(f"**{title}**")
            print("```")
            print("\n".join(lines))
            print("```")
        else:
            if title:
                print(f"### {title}\n")
            print(f"| Player | Cost | {elo_label} | QF% | SF% | F% | W% | EV | EV/Token |")
            print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
            for name in names:
                pd = self.players[name]
                s = player_evs[name]
                f = self._fmt(pd, s)
                evtok = f["ev_tok"] + "*" if name in top_evtok_names else f["ev_tok"]
                print(f"| **{name}** | {pd['cost']} | {f['elo']} "
                      f"| {f['p_qf']} | {f['p_sf']} | {f['p_f']} | {f['p_ch']} "
                      f"| {f['ev']} | {evtok} |")
            print()

    def print_path(self, player_name):
        p_data = self.players[player_name]
        gender  = p_data["gender"]
        p_elo   = p_data["elo"]
        line    = p_data["line"]
        quad    = p_data["quadrant"]
        elo_label = {"elo": "Elo", "gelo": "gElo", "celo": "cElo", "helo": "hElo"}.get(self.elo_col, self.elo_col)

        cost_str = f"Cost {p_data['cost']}  |  " if p_data["is_priced"] else ""
        ev = self.compute_ev(player_name)
        ev_tok_str = f"  |  EV/Tok {ev['ev']/p_data['cost']:.2f}" if p_data["is_priced"] else ""
        header = (f"Path: {player_name}  (line {line}, Q{quad}, {gender})  "
                  f"{elo_label} {round(p_elo)}  |  {cost_str}EV {ev['ev']:.2f}{ev_tok_str}")

        max_rounds = self._max_rounds(gender)
        size = 2 ** max_rounds
        opp_sections = self._bracket_opponent_lines(line, size)

        fb_starts = {1: 1500.0, 2: 1520.0, 3: 1510.0, 4: 1530.0}
        fb_end = 1950.0
        start = fb_starts[quad]
        fb = ([fb_end] if max_rounds == 1
              else [start + (fb_end - start) * i / (max_rounds - 1) for i in range(max_rounds)])

        # round_defs: (label, opp_lines, fallback, scores_at)
        # Rounds 1..max_rounds are real matches (last one = Final).
        # After the Final, append the "W" champion row (no opponent, P(reach) = p_ch).
        # scoring begins at round (1-indexed) = max_rounds - scoring_rounds + 1
        scoring_start = max_rounds - self.scoring_rounds  # 0-based index; rnd > scoring_start → scores
        round_defs = [
            (self._round_label(rnd, max_rounds), opp_sections[rnd - 1], fb[rnd - 1],
             rnd > scoring_start)
            for rnd in range(1, max_rounds + 1)
        ]
        # Champion "W" row — no opponent section, win_p = 1.0
        round_defs.append(("W", [], None, (max_rounds + 1) > scoring_start))

        # Cumulative reach probability, updated each round
        p_reach = 1.0
        rows = []

        for rnd_name, opp_lines, fallback, scores_at in round_defs:
            if rnd_name == "W":
                win_p = 1.0
                opp_str = "—"
                is_bye = False
            else:
                probs = self._section_win_probs(opp_lines, gender)

                if probs:
                    # Bye: all opponents in the section are BYE sentinels (elo 0.0)
                    real_probs = {j: p_j for j, p_j in probs.items()
                                  if self._line_index[(gender, j)]["elo"] > 0.0}
                    if not real_probs:
                        p_reach = p_reach * 1.0  # advance past bye
                        continue
                    is_bye = False
                    top = sorted(real_probs.items(), key=lambda x: -x[1])[:3]
                    win_p = sum(
                        p_j * self.calculate_match_win_prob(p_elo, self._line_index[(gender, j)]["elo"])
                        for j, p_j in probs.items()
                    )
                    opp_parts = []
                    for j, p_j in top:
                        opp_name = next(
                            (n for n, pd in self.players.items() if pd["gender"] == gender and pd["line"] == j),
                            f"line {j}"
                        )
                        opp_elo = self._line_index[(gender, j)]["elo"]
                        opp_parts.append(f"{opp_name} ({round(opp_elo)}, {self._pct(p_j)}%)")
                    opp_str = " / ".join(opp_parts)
                else:
                    is_bye = False
                    win_p = self.calculate_match_win_prob(p_elo, fallback)
                    opp_str = f"unknown (fallback {elo_label} {round(fallback)})"

            p_reach_next = p_reach * win_p
            rnd_pts = 2 * p_reach if scores_at else 0.0

            rows.append([
                rnd_name,
                opp_str,
                self._pct(win_p) + "%",
                self._pct(p_reach) + "%",
                f"{rnd_pts:.2f}" if rnd_pts > 0 else "—",
            ])
            p_reach = p_reach_next

        headers = ["Round", "Opponent(s)  (Elo, P(faces you))", "Win%", "P(reach)", "E[pts]"]

        if self.discord:
            lines_out = self._fixed_table(headers, rows)
            print(f"**{header}**")
            print("```")
            print("\n".join(lines_out))
            print("```")
        else:
            print(f"### {header}\n")
            print("| " + " | ".join(headers) + " |")
            print("| " + " | ".join(["---"] * len(headers)) + " |")
            for row in rows:
                print("| " + " | ".join(str(c) for c in row) + " |")
            print()

    def run_optimization(self):
        if not self.players:
            self.load_data()
        player_evs = {name: self.compute_ev(name) for name in self.players}
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


def fetch_elo_csv(url, rank_col_name):
    from html.parser import HTMLParser
    import urllib.request

    class TableParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.in_table = self.in_row = self.in_cell = False
            self.rows = []
            self.current_row = []
            self.current_cell = []

        def handle_starttag(self, tag, attrs):
            if tag == 'table':
                self.in_table = True
            elif tag == 'tr' and self.in_table:
                self.in_row = True
                self.current_row = []
            elif tag in ('td', 'th') and self.in_row:
                self.in_cell = True
                self.current_cell = []

        def handle_endtag(self, tag):
            if tag == 'table':
                self.in_table = False
            elif tag == 'tr' and self.in_row:
                self.in_row = False
                if self.current_row:
                    self.rows.append(self.current_row)
            elif tag in ('td', 'th') and self.in_cell:
                self.in_cell = False
                self.current_row.append(''.join(self.current_cell).strip().replace('\xa0', ' '))

        def handle_data(self, data):
            if self.in_cell:
                self.current_cell.append(data)

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode('utf-8')

    p = TableParser()
    p.feed(html)

    fieldnames = ['elo_rank', 'player', 'age', 'elo', 'helo_rank', 'helo',
                  'celo_rank', 'celo', 'gelo_rank', 'gelo',
                  'peak_elo', 'peak_month', rank_col_name, 'log_diff']
    rows = []
    for row in p.rows[2:]:
        if not row[0].isdigit():
            continue
        rows.append({
            'elo_rank':      row[0],
            'player':        row[1],
            'age':           row[2],
            'elo':           row[3],
            'helo_rank':     row[5],
            'helo':          row[6],
            'celo_rank':     row[7],
            'celo':          row[8],
            'gelo_rank':     row[9],
            'gelo':          row[10],
            'peak_elo':      row[12],
            'peak_month':    row[13],
            rank_col_name:   row[15],
            'log_diff':      row[16],
        })
    return fieldnames, rows


def update_elo_files():
    print("Fetching men's Elo ratings...", flush=True)
    fieldnames, rows = fetch_elo_csv(
        "https://tennisabstract.com/reports/atp_elo_ratings.html", "atp_rank"
    )
    with open("atp_elo.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} players to atp_elo.csv")

    print("Fetching women's Elo ratings...", flush=True)
    fieldnames, rows = fetch_elo_csv(
        "https://tennisabstract.com/reports/wta_elo_ratings.html", "wta_rank"
    )
    with open("wta_elo.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} players to wta_elo.csv")


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
        )
        optimizer.load_data()

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
