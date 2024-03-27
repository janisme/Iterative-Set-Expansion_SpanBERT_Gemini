# Project2: Information Extraction - Iterative Set Expansion (ISE)

## Purpose

This project is about information extraction on the web, or the task of extracting "structured" information that is embedded in natural language text on the web.
This project implements both a "traditional" information extraction approach (using SpanBERT) that involves multiple steps of data annotation, as well as an approach that reflects the ongoing paradigm shift from multi-step data pipelines with specialized models for extraction tasks to strong "few-shot" learners (using Google's Gemini API).

The objective of this project is to:

   1. Retrieve and parse webpages
   2. Prepare and annotate text on the webpages for subsequent analysis
   3. Extract structured information from the webpages

This is a project is developed in Spring 2024 for the course COMS 6111 Advanced Database Systems taught by Professor Luis Gravano at Columbia University.

Reference: <https://www.cs.columbia.edu/~gravano/cs6111/Proj2/>

## Detail

1. Group members: Dawei Yin, dy2482 & Yenchu Chen, yc4360
2. List of files:
    * `main.py`: This Python script is designed for running an information extraction process using either SpanBERT or Google Gemini for relation extraction from web content, based on a given seed query. It responsible for collecting search results, using spaCy library to extract named entities, and implementing SpenBert.
    * `gemini.py`: This Python script integrates Google's Generative AI (Gemini API) for relation extraction tasks. It defines functions to generate prompts for Gemini, fetch responses from the API, and extract relation tuples.
    * `README.md`: This file contains a description of the project, providing an overview of its purpose and functionality.
    * `requirements.txt`: This file lists all the packages used in the project, ensuring compatibility and ease of setup for other users.
    * `transcript_spanbert.txt`: This file contains the results using SpanBERT classifier with input parameters:`-spanbert 2 0.7 "bill gates microsoft" 10`.Complete in 1 interation with 45 relations extracted.
    * `transcript_gemini.txt`: This file contains the results using Google Gemini API(Gemini 1.0 Pro model) with input parameters:`-gemini 2 0.0 "bill gates microsoft" 10`. Complete in 1 interation with 35 relations extracted.

   The format of transcript follow this reference: <https://www.cs.columbia.edu/~gravano/cs6111/Proj2/Proj2-Transcripts/>.

3. Instruction to run the project:
    Enter into the directory: proj2 where project2.py file is located, then activate the environment.

   ```
   source dbproj/bin/activate
   ```

   Run: python3 project2.py -$model_name “$JSON_API_KEY” “$Engine_ID” “$Gemini API key" $relation $extract_threshold $seed_query $output_len
   For example:

   SpenBERT

   ```
   python3 project2.py -spanbert "AIzaSyBdPoK9zbUZXnDHG4LMMu972zSH7nGdnM8" "56f4e4ae2f4944372" "Gemini-API-key" 2 0.7 "bill gates microsoft" 10
   ```

   Gemini

   ```
   python3 project2.py -gemini "AIzaSyBdPoK9zbUZXnDHG4LMMu972zSH7nGdnM8" "56f4e4ae2f4944372" "Gemini-API-key" 2 0.0 "bill gates microsoft" 10
   ```

   * $model_name: indicating the extraction model used. Options: "-spanbert" for SpanBERT or "-gemini" for Google Gemini.
   * $relation: indicating the type of relation to extract. Options: an integer; where Schools_Attended(1), Work_For(2), Live_In(3), Top_Member_Employees(4).
   * $extract_threshold: indicating the "extraction confidence threshold," which is the minimum extraction confidence that we request for the tuples in the output. Option: a float between 0~1. -gemini ignores this parameter.
   * $seed_query: template of a plausible tuple. A list of words in double quotes corresponding to a plausible tuple for the relation to extract. Example:"bill gates microsoft".
   * $output_len: indicating the number of tuples that are requested in the output. Option: an integer >0.

5. Description of the internal design of this project:

   * High-level Concept[ISE]:
     1. Initialize X (Python dictionary, key is the subject, relation, and object tuple, and value is the confidence rate regards this relationship)
     2. Load the SpanBert pre-trained classifier if SpanBert is specified.
     3. Use the Engine ID with the server we set and the query to get the search result(top-10 webpages), and parse the web content, including remove the excess tab, space, and newline symbols. [search_by_query, parse_response].
     4. Iterates through URLs, and filter out the URL that have seen.
     5. For each unseen URL, extract the content and do extract named entities using spaCy. If the content characters >10000, trim the webpage content to 10000. Determine by the model specified, run SB or run_gemini to do information extraction.[information_extraction, page_extraction]
     6. For each query iteration, modify the X using either SpanBERT method or GEMINI method and print out all the extracted relations. [print_pretty_relations]
     7. If the number of extracted relations is less than the desired output length ($output_len), the process involves concatenating the subject and object from the relation with the highest confidence rate. Furthermore, ensure that this chosen relation hasn't been previously used as a query. Afterward, the process loops back to step 2 for further extraction.
   
     8. If the number of extracted relations > $output_len, teriminate the program. The result relations sets remain all the tuples extractd which may > k.

   * External libraries used:
       1. googleapiclient.discovery: Part of the Google API Client Library for Python. Used to create and work with Google APIs in a Pythonic way. In this case, likely for querying Google Custom Search Engine or other Google services.
       2. sys: Part of Python's standard library. Provides access to some variables used or maintained by the Python interpreter and to functions that interact strongly with the interpreter. Commonly used for manipulating Python runtime environment.
       3. bs4 (BeautifulSoup): A library for parsing HTML and XML documents. Used for web scraping, allowing easy navigation of the parse tree and searching for elements by attributes, tags.
       4. requests: A simple, yet elegant HTTP library. Used for making HTTP requests in Python, such as GET, POST, etc. Process of working with APIs or fetching web content.
       5. spacy: An industrial-strength natural language processing (NLP) library. Provides capabilities for tokenization, entity recognition, dependency parsing, and more.
       6. SpanBERT.spanbert: A library/module possibly related to SpanBERT, an extension of BERT (Bidirectional Encoder Representations from Transformers) focused on improving span-based NLP tasks. Used here for tasks such as entity recognition, relation extraction, or question answering.
       7. SpanBERT.spacy_help_functions: Custom module, containing helper functions to integrate SpanBERT predictions with spaCy document objects or to assist with NLP tasks using spaCy and SpanBERT together.
       8. re:Part of Python's standard library. Provides regular expression matching operations similar to those found in Perl. Used for string searching and manipulation.
       9. time: Part of Python's standard library.Used for timing operations: adding delays.
       10. google.generativeai and google.api_core.exceptions: Part of the Google Cloud client libraries. Used for interacting with Google's generative AI services, such as accessing the Gemini API for text generation. The exceptions module handles API-related errors.

6. Description of SpanBERT and GEMINI method:
   * Information Extraction[information_extraction]:
      1. Check and extract the content from the URL, with removing the excesses space, tabs, or newlines.(page_extraction)
      2. If the content can be used, do Named Entity Recognition by SpaCy and generate doc.
      3. If the SpanBERT is used, feed the generated doc to the SpanBERT, otherwise feed into the GEMINI to extract the relations.

   * SpanBERT[SB]:
   Input Web content:doc that has been processed by spaCy, entities_of_interest and relation_index, threshold:conf, relation sets: acc; update relation sets: acc
      1. Accesses the global spanbert model and sets up variables based on the relation_index to identify the types of entities involved in the desired relation (e.g., PERSON to ORGANIZATION).
      2. Iterates through each sentence in the doc. For every fifth sentence processed, it prints a progress update.
      3. For each sentence, it generates candidate entity pairs (candidate_pairs) that match the specified types of entities relevant to the relation of interest.[create_entity_pairs]
      4. Adjust the subject and object position.
      5. Uses the SpanBERT model to predict the relation for each candidate pair. The prediction includes the type of relation and a confidence score.
      6. Filters out predictions where the relation type does not match the target relation or where the confidence is below a specified threshold (conf).
      7. For predictions that pass the filter, the program checks if the predicted relation is a new extraction or has a higher confidence than a previous prediction for the same entities. If so, the program updates the acc collection with the new prediction.

   * Gemini[run_gemini]:
     Input Web content:doc that has been processed by spaCy, entities_of_interest and relation_index, GEMINI_KEY, relation sets: X; update relation sets: X
        1. Iterates through each sentence in doc. For every fifth sentence processed, it prints a progress update.
        2. Check if the sentence has required named entities pair(s) using spaCy.[create_entity_pairs]
        3. If yes, generate the prompt with the plain-text sentence and designated relation. In this part, the prompt is originally a one-shot learners, and will expand with extracted relations
    to maximum 3-shot learners.[gemini.generate_prompt]
        4. Feed the prompt into Gemini model and get response.[gemini.get_gemini_completion]
        5. Process response. Parse the response into lists of list and if the form of the list is correct update the relation in relation sets. In this step, since we update the dictionary by setting "tuple of relation:1", where 1 is the default confidence for all relationship extracted by Gemini, we do not have to handle the diplicate promblem.[gemini.gemini]
     
      **Note: To avoid termination with Gemini Resource Exhausted, the program will rest for 10 seconds when catch this error.

     ** Note: If the sentence is processced by Gemini but the return relation is not coherent with the defined format, the # of extracted annotaion from Gemini will increase, but no relation will be added to the finel set.

     ** Gemini configuration is set as followed:

      ```
      model_name ="gemini-1.0-pro"
      max_tokens =  100 #0-8192, expecting short response
      temperature = 0.2 #more deterministic, 0-1, control the return format
      top_k = 32 #next-token can,1-40, default =32
      top_p = 1 #select cum. threshold, 0-1, default =1
      ```

6. Testing Key:
   * Google Custom Search Engine JSON API Key

   ```
   AIzaSyDB1xiTbkdr2O8KhnWdHrCJ8jBAfdnxii4
   ```

   * Engine ID

   ```
   56f4e4ae2f4944372
   ```

7. Additional information
   * When handling the output of the Gemini, we should be careful of the return format. Since Gemini generate next token by the largest possibility, the format of the output might varies. We can set lower temperature or more specific prompt to reduce this incoherent.
