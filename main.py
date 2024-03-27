# Reference Call:
# python3 project2.py [-spanbert|-gemini] <google api key> <google engine id> <google gemini api key> <r> <t> <q> <k>
# e.g.
# python3 project2.py -spanbert "AIzaSyBdPoK9zbUZXnDHG4LMMu972zSH7nGdnM8" "56f4e4ae2f4944372" "123" 2 0.7 "bill gates microsoft" 10
# import argparse
from googleapiclient.discovery import build
import sys
from bs4 import BeautifulSoup
import requests
import spacy
from SpanBERT.spanbert import SpanBERT
from SpanBERT.spacy_help_functions import *
import re
from gemini import gemini

nlp = spacy.load("en_core_web_lg")


spanbert = None
CX = "56f4e4ae2f4944372"  # engine ID
KEY = "xxxxx"  # Key_ code injection

relation_map = {
    0: "per:schools_attended",
    1: "per:employee_of",
    2: "per:cities_of_residence",
    3: "org:top_members/employees",
}
target_relation = ""

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.google.com/",
    "DNT": "1",  # Do Not Track Request Header
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "max-age=0",
}

entity_of_interests_lst = [
    ("PERSON", "ORGANIZATION"),
    ("PERSON", "ORGANIZATION"),
    ("PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"),
    ("ORGANIZATION", "PERSON"),
]


def parse_response(response):
    """title, URL, and description"""
    r = {}
    r["title"] = response["title"]
    r["url"] = response["link"]
    if "snippet" in response:
        r["summary"] = response["snippet"]
    else:
        r["summary"] = None
    return r


def search_by_query(query, engine_id, engine_key):
    service = build("customsearch", "v1", developerKey=engine_key)
    response = (
        service.cse()
        .list(
            q=query,
            cx=engine_id,
        )
        .execute()
    )
    results = []
    html_result = []
    non_html_idxs = set()

    for i, r in enumerate(response["items"]):
        if "fileFormat" in r:
            non_html_idxs.add(i)
        else:
            html_result.append(parse_response(r))
        results.append(parse_response(r))
    # print(results)
    return results, html_result, non_html_idxs


def page_extraction(url):
    print(f"{url}")
    try:
        response = requests.get(url=url, headers=headers, timeout=60)  # timeout 60s
        if response.status_code != 200:
            print(f"Request to {url} failed with status code {response.status_code}")
            return None, False
    except Exception as e:
        print(e)
        return None, False

    soup = BeautifulSoup(response.text, "html.parser")
    # for script in soup(["script", "style",'[document]', 'head', 'title']):
    #     script.extract()    # rip it out
    text = soup.get_text()
    text = re.sub(r"[\n\t\s]+", " ", text)
    lines = (line.strip() for line in text.splitlines())
    text.strip()
    text = " ".join(lines)

    print(f"Webpage length (num characters): {len(text)}")
    # truncate if necessary
    if len(text) > 10000:
        print(f"Trimming webpage content from {len(text)} to {10000} characters")
        text = text[:10000]
    return text, True


def information_extraction(url, relation_index, mode, conf, acc, query, GEMINI_KEY):
    # define entities set by relation_index
    entities_of_interest = entity_of_interests_lst[relation_index]

    # method: [-spanbert|-gemini]
    content, ok = page_extraction(url)
    if not ok:
        return False
    doc = nlp(content)
    # spanbert or gemini for doc
    if mode == "-spanbert":
        SB(doc, relation_index, conf, acc, entities_of_interest)
    elif mode == "-gemini":
        run_gemini(GEMINI_KEY, doc, relation_index, acc, entities_of_interest)
    return True


def run_gemini(GEMINI_KEY, doc, relation_index, X, entities_of_interest):
    ori_len = len(X.keys())

    print(
        f"Extracted {len(list(doc.sents))} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, run gemini ..."
    )
    anno = 0
    g_count = 0
    for i, sentence in enumerate(doc.sents):
        # report process
        if (i + 1) % 5 == 0:
            print(f"Processed {i+1} / {len(list(doc.sents))} sentences")

        # check if the content contains tag neede in relation
        sentence_entity_pairs = create_entity_pairs(sentence, entities_of_interest)
        if len(sentence_entity_pairs) == 0:
            continue
        else:
            # print("\tProcess with Gemini. Sentence: {}".format(sentence))
            anno += 1
            count, prompt = gemini(GEMINI_KEY, relation_index, X, sentence, g_count)
            g_count += count

    # print one example prompt
    if g_count > 0:
        print(f"\nLast prompt={prompt}")
    print(
        f"\nExtracted annotations for  {anno}  out of total  {len(list(doc.sents))}  sentences from spaCy, {g_count} from Gemini."
    )
    print(
        f"Relations extracted from this website: {len(X.keys()) - ori_len} (Overall: {len(X.keys())})\n\n"
    )

    return X


def SB(doc, relation_index, conf, acc, entities_of_interest):
    global spanbert
    # subject:: PERSON object:: {'ORGANIZATION'}
    sub = entity_of_interests_lst[relation_index][0]
    objective = set(entity_of_interests_lst[relation_index][1:])
    ext_ct = 0
    ext_total = 0
    ext_st_ct = 0
    print(
        f"Extracted {len(list(doc.sents))} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ..."
    )
    for i, sentence in enumerate(doc.sents):
        # ents = get_entities(sentence, entities_of_interest)
        # print("sentence:::", sentence)
        # print("spaCy extracted entities: {}".format(ents))
        if (i + 1) % 5 == 0:
            print(f"Processed {i+1} / {len(list(doc.sents))} sentences")
        candidate_pairs = []
        sentence_entity_pairs = create_entity_pairs(sentence, entities_of_interest)
        for ep in sentence_entity_pairs:
            e1, e2 = ep[1], ep[2]
            if e1[1] == sub and e2[1] in objective:
                candidate_pairs.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})
            if e2[1] == sub and e1[1] in objective:
                candidate_pairs.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})

        # print("candidate_pairs",candidate_pairs)
        if len(candidate_pairs) == 0:
            continue

        relation_preds = spanbert.predict(
            candidate_pairs
        )  # get predictions: list of (relation, confidence) pairs
        for ex, pred in list(zip(candidate_pairs, relation_preds)):
            relation = pred[0]
            if relation == "no_relation" or relation != relation_map[relation_index]:
                continue
            print("\n\t\t=== Extracted Relation ===")
            ext_total += 1
            print("\tSentence: {}".format(sentence))
            subj = ex["subj"][0]
            obj = ex["obj"][0]
            confidence = pred[1]
            print(
                "\t\tRelation: {} (Confidence: {:.3f})\nSubject: {}\tObject: {}".format(
                    relation, confidence, subj, obj
                )
            )
            if confidence > conf:

                if acc[(subj, relation, obj)] < confidence:
                    ext_ct += 1
                    acc[(subj, relation, obj)] = confidence
                    print("\t\tAdding to set of extracted relations")
                else:
                    print(
                        "\t\tDuplicate with lower confidence than existing record. Ignoring this."
                    )
            else:
                print(
                    "\t\tConfidence is lower than threshold confidence. Ignoring this."
                )
            print("\t\t==========")

    print(
        f"Extracted annotations for  {ext_st_ct}  out of total  {len(list(doc.sents))}  sentences"
    )
    ext_st_ct += 1
    print(f"Relations extracted from this website: {ext_ct} (Overall: {ext_total})")
    return acc


def ISE(query, mode, relation_index, conf, k, engine_id, engine_key, GEMINI_KEY):
    # iterative set expansion
    X = defaultdict(int)  # (sub,relation,obj): conf
    seen_url = set()
    used_query = set()
    used_query.add(query)
    # load pretrained only invoke the spanbert
    global spanbert
    if mode == "-spanbert":
        spanbert = SpanBERT("./SpanBERT/pretrained_spanbert")
    cur_query = query
    iter_count = 0
    while len(X.keys()) < k:

        print(f"=========== Iteration: {iter_count} - Query: {cur_query} =============")
        iter_count += 1

        result, _, _ = search_by_query(cur_query, engine_id, engine_key)
        for i, r in enumerate(result):
            print(f"URL ({i+1}/10) :", end=" ")
            url = r["url"]
            if url in seen_url:
                print("Skip visited URL...")
                continue
            seen_url.add(url)
            # process url
            information_extraction(
                url, relation_index, mode, conf, X, query, GEMINI_KEY
            )

        # print X for each iterations
        print_pretty_relations(X)

        # generate new query
        top = 0
        ord_candidate_tuples = sorted(X.items(), key=lambda x: x[1], reverse=True)
        candidate_query = (
            f"{ord_candidate_tuples[top][0][0]} {ord_candidate_tuples[top][0][2]}"
        )
        while candidate_query in used_query:
            top += 1
            candidate_query = (
                f"{ord_candidate_tuples[top][0][0]} {ord_candidate_tuples[top][0][2]}"
            )
            if top >= len(ord_candidate_tuples):
                raise "should not happen"
        cur_query = candidate_query
        used_query.add(cur_query)

    print("Total # of iterations = ", iter_count)

    # return top all valid results
    ord_candidate_tuples = sorted(X.items(), key=lambda x: x[1], reverse=True)


def print_pretty_relations(X):
    global target_relation
    sorted_relations = sorted(X.items(), key=lambda item: item[1], reverse=True)
    print("=" * 80)
    print(f"ALL RELATIONS for {target_relation} ({len(sorted_relations)})")
    print("=" * 80)

    for (subject, _, obj), confidence in sorted_relations:
        print(
            "Confidence: {:.7f} \t| Subject: {} \t| Object: {}".format(
                confidence, subject, obj
            )
        )


def main():
    if len(sys.argv) != 9:  # Check if the correct number of arguments are provided
        print(
            "Usage: <mode> <google_api_key> <google_engine_id> <google_gemini_api_key> <r> <t> <q> <k>"
        )
        sys.exit(1)

    mode = sys.argv[1]
    google_api_key = sys.argv[2]
    google_engine_id = sys.argv[3]
    GEMINI_KEY = sys.argv[4]
    r = int(sys.argv[5])
    t = float(sys.argv[6])
    q = sys.argv[7]
    k = int(sys.argv[8])
    global target_relation
    target_relation = relation_map[r - 1]

    print("Parameters:")
    print("Google API Key:", google_api_key)
    print("Google Engine ID:", google_engine_id)
    print("Google Gemini API Key:", GEMINI_KEY)
    print("Mode:", mode)
    print("Relation:", relation_map[r - 1])  # 1-4
    print("Seed Query:", q)
    print("Number of Tuples:", k)
    if mode == "-spanbert":
        global spanbert
        print("Threshold:", t)

    ISE(q, mode, r - 1, t, k, google_engine_id, google_api_key, GEMINI_KEY)


if __name__ == "__main__":
    main()
