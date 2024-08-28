import pandas as pd
from Levenshtein import ratio

# File path
file = "/Users/fmys/Documents/Data/master_thesis/0_MultiMedQA_final/0_MultiMedQA.csv"

# Column names
column_names = [
    "Question", "Correct Answer", "Correct Answer ID", "Answer GPT4o 0-shot",
    "Answer GPT4turbo 0-shot", "Answer GPT3.5Turbo 0-shot", "Answer GPT3.5TurboFinetuned 0-shot",
    "Answer GPT4o 5-shot", "Answer GPT4turbo 5-shot", "Answer GPT3.5Turbo 5-shot",
    "Answer GPT3.5TurboFinetuned 5-shot", "Answer Options", "Dataset", "Unique ID",
    "0-shot Prompt", "5-shot Prompt", "Calibration Prompt 0-shot GPT4o",
    "Answer Calibration GPT4o 0-shot", "Answer Status GPT4o 0shot",
    "Memorization q1", "Memorization q2", "Memorization Prompt", "Memorization Answer GPT4o"
]

# Read the CSV file
df = pd.read_csv(file, usecols=column_names)

# Function to calculate the Levenshtein ratio
def calculate_levenshtein_ratio(row):
    return ratio(str(row["Memorization q2"]), str(row["Memorization Answer GPT4o"]))

# Apply the function to each row
df["Levenshtein_ratio"] = df.apply(calculate_levenshtein_ratio, axis=1)

# Define near-exact match threshold
near_exact_threshold = 0.95

# Calculate percentage of near-exact matches for each dataset
result = df.groupby("Dataset").apply(lambda x: (x["Levenshtein_ratio"] >= near_exact_threshold).mean() * 100)
result = result.reset_index(name="Percentage of Near-Exact Matches")

# Print the result
print(result)

# Print results in a more detailed format
print("\nPercentage of data points with near-exact matches for each dataset:")
for _, row in result.iterrows():
    print(f"{row['Dataset']}: {row['Percentage of Near-Exact Matches']:.2f}%")