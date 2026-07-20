import csv
from html.parser import HTMLParser
import urllib.request


class _TableParser(HTMLParser):
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


def fetch_elo_csv(url, rank_col_name):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode('utf-8')

    p = _TableParser()
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
