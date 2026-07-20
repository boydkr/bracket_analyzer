def pct(v):
    """Format a probability (0–1) as a percentage string.
    <1%: 2 decimal places, no leading zero (.04); <10%: 1 decimal (8.2); else integer (62)."""
    p = v * 100
    if p < 1:
        return f"{p:.2f}".lstrip("0")
    elif p < 10:
        return f"{p:.1f}"
    else:
        return f"{p:.0f}"


def fmt_player(pd, s, elo_col="elo"):
    """Return consistently formatted display values for a player."""
    elo_label = {"elo": "elo", "gelo": "gelo", "celo": "celo", "helo": "helo"}.get(elo_col, elo_col)
    d = {
        "elo":      str(round(pd["elo"])),
        "elo_label": elo_label,
        "p_qf":     pct(s['p_qf']),
        "p_sf":     pct(s['p_sf']),
        "p_f":      pct(s['p_f']),
        "p_ch":     pct(s['p_ch']),
        "ev":       f"{s['ev']:.2f}",
        "ev_tok":   f"{s['ev']/pd['cost']:.2f}",
    }
    if s.get("draw_eff") is not None:
        d["draw_eff"] = f"{s['draw_eff']:.2f}"
    else:
        d["draw_eff"] = "—"
    return d


def fixed_table(headers, rows):
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
