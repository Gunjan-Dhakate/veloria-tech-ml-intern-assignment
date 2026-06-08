import os
import re
import logging
import requests
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO
from urllib.parse import urljoin

# =====================================================
# CONFIGURATION
# =====================================================

BASE_DIR = r"C:\Users\gunja\Downloads\veloria-ml-assignment_1\veloria-tech-ml-intern-assignment"

OUTPUT_FILE = os.path.join(BASE_DIR, "match_data.csv")
LOG_FILE = os.path.join(BASE_DIR, "scraper.log")

BASE_URL = "https://www.howstat.com"

MATCH_LIST_URL = (
    "https://www.howstat.com/Cricket/Statistics/IPL/"
    "MatchScores.asp?q=1&s=2026&t=CSK"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.howstat.com/Cricket/Statistics/IPL/MatchScores.asp?q=1&s=2026&t=CSK",
    "Connection": "keep-alive"
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

# =====================================================
# LOGGING
# =====================================================

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =====================================================
# REQUEST
# =====================================================

def get_soup(url):

    response = SESSION.get(
        url,
        timeout=20
    )

    response.raise_for_status()

    return BeautifulSoup(
        response.text,
        "html.parser"
    )


# =====================================================
# TOP SCORER EXTRACTION
# =====================================================

def get_top_scorer(scorecard_url):

    try:

        response = SESSION.get(
            scorecard_url,
            timeout=20
        )

        response.raise_for_status()

        tables = pd.read_html(
            StringIO(response.text),
            flavor="lxml"
        )

        best_player = ""
        best_runs = 0

        for table in tables:

            cols = [str(c).strip() for c in table.columns]
            
            runs_col = None
            player_col = None
            
            for col in cols:
                if col == "R" or col == "Runs" or col.endswith(" R") or col.endswith("Runs"):
                    runs_col = col
                    break
            
            if runs_col:
                for idx, row in table.iterrows():

                    try:

                        runs_str = str(row[runs_col]).strip()
                        
                        if runs_str and runs_str != "nan" and not runs_str.startswith("R"):
                            runs = int(float(runs_str.split()[0]))

                            if runs > best_runs:
                                
                                player_name = str(row.iloc[0]).strip()
                                
                                if player_name and player_name != "nan" and len(player_name) > 1:
                                    best_runs = runs
                                    best_player = player_name

                    except (ValueError, IndexError):
                        pass
        
        if not best_player:
            soup = BeautifulSoup(response.text, "html.parser")
            page_text = soup.get_text(" ", strip=True)
            
            patterns = [
                r"([A-Z][A-Za-z'.\-]+(?:\s+[A-Z][A-Za-z'.\-]+)+)\s+(\d{1,3})\s+(?:runs|R|Runs|not out|retired)",
                r"([A-Z][A-Za-z'.\-]+(?:\s+[A-Z][A-Za-z'.\-]+)+)\s+(\d{1,3})\*?(?:\s|$)"
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, page_text)
                for player, runs in matches:
                    try:
                        runs_int = int(runs)
                        if 0 < runs_int < 300 and runs_int > best_runs:
                            best_player = player.strip()
                            best_runs = runs_int
                    except ValueError:
                        pass

        return best_player, best_runs

    except Exception as e:

        logging.error(
            f"Top scorer extraction failed: {e}"
        )

        return "Unknown", 0


# =====================================================
# MATCH TABLE SCRAPER
# =====================================================

def scrape_matches():

    soup = get_soup(MATCH_LIST_URL)

    tables = pd.read_html(
        StringIO(str(soup)),
        flavor="lxml"
    )

    match_table = None

    for table in tables:

        cols = [str(c) for c in table.columns]

        if (
            "Team" not in cols
            and len(table) > 0
            and all(str(c).strip() == "" or isinstance(c, int) for c in table.columns)
        ):
            header_row = table.iloc[0].astype(str).str.strip().tolist()
            if "Team" in header_row and "Versus" in header_row and "Venue" in header_row:
                table = table[1:].copy()
                table.columns = header_row
                table.reset_index(drop=True, inplace=True)
                cols = [str(c) for c in table.columns]

        if (
            "Team" in cols
            and "Versus" in cols
            and "Venue" in cols
        ):

            match_table = table
            break

    if match_table is None:
        raise ValueError(
            "Match table not found."
        )

    records = []

    scorecard_links = []

    for a in soup.find_all("a", href=True):

        href = a["href"]

        if "MatchScorecard.asp" in href:

            if not href.startswith("/"):
                href = "/Cricket/Statistics/IPL/" + href
            
            scorecard_links.append(
                urljoin(BASE_URL, href)
            )

    scorecard_links = scorecard_links[:10]

    for idx, row in match_table.head(10).iterrows():

        try:

            team_1 = row["Team"]
            team_2 = row["Versus"]
            venue = row["Venue"]
            result = row["Result"]
            match_date = row["Scorecard"]

            winner = result.split(" won")[0]

            top_scorer = "Unknown"
            top_runs = 0

            if idx < len(scorecard_links):

                scorecard_url = scorecard_links[idx]
                top_scorer, top_runs = (
                    get_top_scorer(scorecard_url)
                )

            record = {
                "match_date": match_date,
                "team_1": team_1,
                "team_2": team_2,
                "venue": venue,
                "winner": winner,
                "top_scorer": top_scorer,
                "top_scorer_runs": top_runs
            }

            records.append(record)

            logging.info(
                f"Scraped: {team_1} vs {team_2}"
            )

        except Exception as e:

            logging.error(
                f"Row scrape failed: {e}"
            )

    return pd.DataFrame(records)


# =====================================================
# VALIDATION
# =====================================================

def validate_dataframe(df):

    df.drop_duplicates(inplace=True)

    df.fillna("Unknown", inplace=True)

    return df


# =====================================================
# MAIN
# =====================================================

def main():

    print("=" * 60)
    print("VELORIA TECH - TASK 1")
    print("=" * 60)

    df = scrape_matches()

    df = validate_dataframe(df)

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nCSV Created:")
    print(OUTPUT_FILE)

    print(
        f"\nRows Saved: {len(df)}"
    )

    print("\nPreview:\n")
    print(df.head())

    logging.info(
        f"CSV saved successfully "
        f"with {len(df)} rows"
    )


if __name__ == "__main__":
    main()