import pandas as pd
import random

# Set the random seed
random.seed(42)

# Read the CSV file
path = "/Users/fmys/Documents/Data/master_thesis/0_MultiMedQA_final/0_MultiMedQA.csv"
df = pd.read_csv(path)


# Function to create a single example
def create_example(row):
    return f"""
Question: {row['Question']}
Answer Options: {row['Answer Options']}
Correct Answer ID: {row['Correct Answer ID']}
"""


# Function to create the 5-shot prompt
def create_5shot_prompt(examples, question, answer_options):
    examples_5 = "\n".join(examples)
    return f"""
In the following you will receive 5 answered examples and then a question and a list of answer options.

5 Examples:
{examples_5}


Question to be answered now:
{question}

Answer Options:
{answer_options}

It is extremely important that you only answer with one of the provided answer options.
This means you should only answer with the Answer ID, i.e. a single letter, single number or yes/no/maybe according to the provided Answer Options.
"""


# Function to generate 5-shot prompt for each row
def generate_5shot_prompt(row, df):
    dataset = row['Dataset']
    subset = df[df['Dataset'] == dataset]

    # Exclude the current row from the subset
    subset = subset[subset['Unique ID'] != row['Unique ID']]

    # Randomly select 5 examples
    sample = subset.sample(n=min(5, len(subset)))

    examples = [create_example(r) for _, r in sample.iterrows()]

    return create_5shot_prompt(examples, row['Question'], row['Answer Options'])


# Apply the function to each row
df['5-shot Prompt'] = df.apply(lambda row: generate_5shot_prompt(row, df), axis=1)

# Save the updated DataFrame back to CSV
df.to_csv(path, index=False)

print("5-shot prompts have been generated and added to the CSV file.")