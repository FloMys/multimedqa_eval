import pandas as pd

# Load the CSV file
file = "/Users/fmys/Documents/Data/master_thesis/0_MultiMedQA_final/0_MultiMedQA.csv"
column_names = [
    "Question", "Correct Answer", "Correct Answer ID", "Answer GPT4o 0-shot", "Answer GPT4turbo 0-shot",
    "Answer GPT3.5Turbo 0-shot", "Answer GPT3.5TurboFinetuned 0-shot", "Answer GPT4o 5-shot",
    "Answer GPT4turbo 5-shot", "Answer GPT3.5Turbo 5-shot", "Answer GPT3.5TurboFinetuned 5-shot",
    "Answer Options", "Dataset", "Unique ID", "0-shot Prompt", "5-shot Prompt",
    "Calibration Prompt 0-shot GPT4o", "Answer Calibration GPT4o 0-shot", "Answer Status GPT4o 0shot",
    "Memorization q1", "Memorization q2", "Memorization Prompt", "Memorization Answer GPT4o"
]

df = pd.read_csv(file, usecols=column_names)

# Function to calculate the number of matches (M) between two strings
def number_of_matches(s1, s2):
    return sum(1 for a, b in zip(s1, s2) if a == b)

# Function to calculate the Levenshtein ratio as per the paper
def calculate_levenshtein_ratio(row):
    s1 = str(row["Memorization q2"])
    s2 = str(row["Memorization Answer GPT4o"])
    if len(s1) == 0 and len(s2) == 0:
        return 1.0  # Both strings are empty, consider them as a perfect match
    M = number_of_matches(s1, s2)
    return (2 * M) / (len(s1) + len(s2))

# Apply the function to each row, handling NaN values by filling them with empty strings
df["Memorization q2"] = df["Memorization q2"].fillna("").astype(str)
df["Memorization Answer GPT4o"] = df["Memorization Answer GPT4o"].fillna("").astype(str)

df["Levenshtein Ratio"] = df.apply(calculate_levenshtein_ratio, axis=1)

# Define near-exact match threshold
threshold = 0.95

# Calculate percentage of near-exact matches for each dataset
result = df.groupby("Dataset").apply(lambda x: (x["Levenshtein Ratio"] >= threshold).mean() * 100)
result = result.reset_index(name="Percentage of Near-Exact Matches")

# Print the result
print(result)
