"""
https://github.com/pubmedqa/pubmedqa
https://pubmedqa.github.io/

PubMedQA has 1k expert labeled (PQA-L), 61.2k unlabeled (PQA-U) and 211.3k artificially generated QA (PQA-A) instances.
For my benchmark I will only be using PQA-L!

Original Data Fields
1. PMID (string): The PubMed ID of the article, used as the unique identifier for each example
2. QUESTION (string): The question being asked about the article
3. CONTEXTS (list of strings): A list of relevant excerpts from the article, providing context to answer the question. Each excerpt is labeled with a section header like BACKGROUND, METHODS, RESULTS, etc.
4. LABELS (list of strings): The section labels corresponding to each excerpt in the CONTEXTS field
5. MESHES (list of strings): MeSH (Medical Subject Headings) terms associated with the article
6. YEAR (string): The publication year of the article
7. reasoning_required_pred (string): A binary prediction of whether answering the question requires reasoning over multiple excerpts in CONTEXTS. Values are "yes", "no" or "maybe".
8. reasoning_free_pred (string): A binary prediction of whether the question can be answered from a single excerpt in CONTEXTS without reasoning. Values are "yes" or "no".
9. final_decision (string): The final decision on whether answering the question requires reasoning, considering both reasoning_required_pred and reasoning_free_pred. Values are "yes", "no" or "maybe".
10. LONG_ANSWER (string): A long form answer to the question, typically a few sentences summarizing the key information from the CONTEXTS excerpts

Transformed into (all datafields are strings):
"Question": question to be answered together with the provided context which is labeled as such
"Correct Answers": Correct answer to the question. Here this is yes / no / maybe from test_ground_truth.json.
"Correct Answer IDs": Only relevant if answers are multiple-choice otherwise leave empty. Here this is yes / no / maybe from test_ground_truth.json
"Answer Options": Only relevant if answers are multiple-choice otherwise leave empty. Here this is yes / no / maybe.
"Dataset": Name of the dataset this question is from ("PubMedQA - PQA-L with ground truth labels - 400 samples")
"Original ID": Originally used PMID (PubMed ID) of the article

"""