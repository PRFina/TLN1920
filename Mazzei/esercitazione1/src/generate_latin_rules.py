import requests as req
from bs4 import BeautifulSoup
from pathlib import Path


html_docs = [("VERB", "https://en.wiktionary.org/wiki/Category:Latin_verb-forming_suffixes"),
             ("NOUN",  "https://en.wiktionary.org/wiki/Category:Latin_noun-forming_suffixes"),
             ("ADJ",  "https://en.wiktionary.org/wiki/Category:Latin_adjective-forming_suffixes"),
             ("ADV",  "https://en.wiktionary.org/wiki/Category:Latin_adverb-forming_suffixes")]

# output path where to store rules file
suffix_filepath = Path("../data/latin_derivational_suffixes_rules.txt")

if __name__ == "__main__":
    print("Extracting rules")
    with suffix_filepath.open("w") as suffix_file:
        rules = []
        for POS_tag, url in html_docs:
            r = req.get(url)
            soup = BeautifulSoup(r.text, 'html.parser')

            for suffix_el in soup.body.find("div", {"id":"mw-pages"}).find_all("li"):
                suffix = suffix_el.text.strip("-")
                print("pattern discovered")
                if  len(suffix) >= 3 and len(suffix) <= 6: # simple heuristic to filter out noise
                    rules.append(f".*{suffix}$,{POS_tag}")

        print(f"dumping {len(rules)} rules into {str(suffix_filepath.resolve())}")
        suffix_file.writelines("\n".join(rules)) 