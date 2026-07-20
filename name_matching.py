import unicodedata


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
