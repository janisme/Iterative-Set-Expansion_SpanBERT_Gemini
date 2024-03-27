# from collections import defaultdict
# import os
import re
import time
import google.generativeai as genai
import google.api_core.exceptions

debug = False

relation_name = {
    0: "Schools_attended",
    1: "Work_For",
    2: "Live_in",
    3: "Top_Member_Employees",
}


# generage prompt with task sentence and few shot(<3) from extracted result
def generate_prompt(relation_index, X, sentence):
    restriction = {
        0: ["'Person'", "'Organization'"],
        1: ["'Person'", "'Organization'"],
        2: [
            "'Person'",
            "either of 'LOCATION', 'CITY', 'STATE_OR_PROVINCE', or 'COUNTRY'",
        ],
        3: ["'Organization'", "'Person'"],
    }
    definition = {
        0: f"a {restriction[0][0]} has attended {restriction[0][1]} for education",
        1: f"a {restriction[1][0]} is employed by or works for a {restriction[1][1]}",
        2: f"a {restriction[2][0]} has lived in {restriction[2][1]}",
        3: f"a {restriction[3][1]} is a top member or employee of an {restriction[3][0]}",
    }

    one_shot = {
        0: "Jeff Bezos, Schools_Attended, Princeton University",
        1: "Alec Radford, Work_For, OpenAI",
        2: "Mariah Carey, Live_In, New York City",
        3: "Nvidia, Top_Member_Employees, Jensen Huang",
    }

    # max shot = 3
    shot = 0
    example = ""
    for key in X:
        if X[key] == 1 and shot < 2:
            example += f"""; {key[0]}, {key[1]}, {key[2]}"""
            shot += 1

    prompt = f"""
    Task: Identify entities labeled as {restriction[relation_index][0]} and {restriction[relation_index][1]} \
and analyze the relations between those entities to extract all explicit {relation_name[relation_index]} relations in the sentence.
    Sentence : {sentence}
    Definition: A {relation_name[relation_index]} relation indicates that {definition[relation_index]}.
    Return format: represent relation in the format of [{restriction[relation_index][0]},{relation_name[relation_index]},{restriction[relation_index][1]}], and use ";" to seperate each relationship.
    * If there is no relationship exists return NULL.
    * Object must be proper nouns, not pronouns not NULL.
    * Subject must be proper nouns, not pronouns not NULL.
    * example: {one_shot[relation_index]}{example}
    """

    return prompt


# get gemini return
def get_gemini_completion(
    GEMINI_KEY, prompt, model_name, max_tokens, temperature, top_k, top_p
):
    # apply api key
    # GEMINI_KEY = "AIzaSyArX0np_auOKj09EGr0OKz8hos0BoM06cs" #GapiKey
    genai.configure(api_key=GEMINI_KEY)

    # Initialize a generative model
    model = genai.GenerativeModel(model_name)

    # Configure the model with desired parameters
    generation_config = genai.types.GenerationConfig(
        max_output_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k
    )

    # Generate a response
    try:
        response = model.generate_content(prompt, generation_config=generation_config)
        # print(response.text)
    except google.api_core.exceptions.ResourceExhausted as e:
        # retry after 10 seconds
        print("google.api_core.exceptions.ResourceExhausted wait 10s retry again")
        time.sleep(10)
        return get_gemini_completion(
            GEMINI_KEY, prompt, model_name, max_tokens, temperature, top_k, top_p
        )

    return response.text


def gemini(GEMINI_KEY, relation_index, X, sentence, g_count):
    # get prompt
    prompt = generate_prompt(relation_index, X, sentence)

    # get_gemini_completion
    # gemini model specify
    model_name = "gemini-1.0-pro"
    max_tokens = 100  # 0-8192
    temperature = 0.2  # more deterministic, 0-1
    top_k = 32  # next-token can,1-40, default =32
    top_p = 1  # select cum. threshold, 0-1, default =1

    # get candidate
    if debug:
        print(prompt)
    extracted_string = get_gemini_completion(
        GEMINI_KEY, prompt, model_name, max_tokens, temperature, top_k, top_p
    )

    if extracted_string == "NULL":
        return 0, prompt

    if debug:
        print(extracted_string)

    # process cand verify cand. then store into X
    lines = re.split(";|\n", extracted_string)
    first = True
    for line in lines:
        clean_line = line.strip("- ").strip("[]")

        # [subject, relation, object]
        elements = [
            element.strip(" ").strip('"').strip("'").strip(";").strip("[]")
            for element in clean_line.split(",")
        ]
        if debug:
            print(elements)
        if len(elements) < 3:
            continue
        elif first:
            print("\n\t\t=== Extracted Relation ===")
            print(f"\t\tSentence:{sentence}")
            first = False

        X[(elements[0], elements[1], elements[2])] = 1

        print("\n\t\tSubject: {} ;\tObject: {}".format(elements[0], elements[2]))
        print("\t\tAdding to set of extracted relations...")
    if debug:
        print(prompt)
    if not first:
        print("\t\t==========")

    return 1, prompt


def main():
    # testing
    # extracted_string ="fd fd sf"
    # sentence = f"""More:USWNT soccer: How can I stream the ticker-tape parade in NYC's Canyon of Heroes?In an energetic Instagram story Rapinoe posted celebrating the win, one shot shows the Record Searchlight's Monday cover page that featured a full-page picture of her after the historic win."""
    # nlp = spacy.load("en_core_web_lg")
    # doc = nlp("Bill Gates stepped down as chairman of Microsoft in February 2014 and assumed a new post as technology adviser to support the newly appointed CEO Satya Nadella. Henry works in Tesla.")

    # label_dict ={}
    # for ent in doc.ents:
    #     label_dict[ent.text] = ent.label_

    # # print(label_dict["Bill Gates"])

    # X = defaultdict(int)
    # #X[('Bill Gates','', 'Microsoft')] = 1
    # init_query = f"""megan rapinoe redding"""

    # extracted_tuples = gemini(1, X, sentence,init_query,label_dict)
    # print(extracted_tuples)
    pass


if __name__ == "__main__":
    main()
