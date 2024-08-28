import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Levenshtein import ratio

# Load the CSV file
file = "/Users/fmys/Documents/Data/master_thesis/0_MultiMedQA_final/0_MultiMedQA.csv"
df = pd.read_csv(file)

# Count total rows and null values before dropping
total_rows = len(df)
null_counts_before = df[["Memorization q2", "Memorization Answer GPT4o"]].isnull().sum()

# Remove rows with None or NaN in the specified columns
df_cleaned = df.dropna(subset=["Memorization q2", "Memorization Answer GPT4o"])

# Count rows after dropping
rows_after = len(df_cleaned)

# Calculate dropped rows for each column
dropped_rows = {
    "Memorization q2": null_counts_before["Memorization q2"],
    "Memorization Answer GPT4o": null_counts_before["Memorization Answer GPT4o"],
    "Total": total_rows - rows_after
}

print("Rows dropped due to NaN or None values:")
for col, count in dropped_rows.items():
    print(f"{col}: {count}")

# Function to calculate the Levenshtein ratio using the standard library
def calculate_levenshtein_ratio(row):
    s1 = str(row["Memorization q2"])
    s2 = str(row["Memorization Answer GPT4o"])
    return round(ratio(s1, s2), 4)  # Round to 4 decimal places

# Apply the function to each row and add the Levenshtein Ratio column
df_cleaned["Levenshtein Ratio"] = df_cleaned.apply(calculate_levenshtein_ratio, axis=1)

# Save the updated DataFrame with the new Levenshtein Ratio column
df_cleaned.to_csv(file, index=False)
print(f"Updated CSV with Levenshtein Ratio has been saved to: {file}")

# Function to print examples for a given dataset and threshold
# Function to print examples for a given dataset and threshold
def print_examples(dataset, threshold, n=5):
    examples = df_cleaned[(df_cleaned["Dataset"] == dataset) & (df_cleaned["Levenshtein Ratio"] >= threshold)]
    print(f"\nExamples for {dataset} with Levenshtein Ratio >= {threshold}:")
    for i, (_, row) in enumerate(examples.iterrows()):
        if i >= n:
            break
        print(f"Example {i+1}:")
        print(f"Unique ID: {row['Unique ID']}")
        print(f"Q1: {row['Memorization q1']}")
        print(f"Original Q2: {row['Memorization q2']}")
        print(f"GPT-4o Generated Q2: {row['Memorization Answer GPT4o']}")
        print(f"Answer Options: {row['Answer Options']}")
        print(f"Levenshtein Ratio: {row['Levenshtein Ratio']}")
        print()

# Print examples for each dataset
for dataset in df_cleaned["Dataset"].unique():
    print_examples(dataset, 0.9)

# Define near-exact match threshold
threshold = 0.95

# Calculate percentage of near-exact matches for each dataset
result = df_cleaned.groupby("Dataset", group_keys=False).apply(lambda x: pd.Series({
    "Percentage of Near-Exact Matches": (x["Levenshtein Ratio"] >= threshold).mean() * 100
}))
result = result.reset_index()

# Rename datasets
dataset_mapping = {
    "MMLU - test - anatomy": "MMLU - anatomy",
    "MMLU - test - clinical knowledge": "MMLU - clinical knowledge",
    "MMLU - test - college biology": "MMLU - college biology",
    "MMLU - test - college medicine": "MMLU - college medicine",
    "MMLU - test - medical genetics": "MMLU - medical genetics",
    "MMLU - test - professional medicine": "MMLU - professional medicine",
    "MedMCQA - dev": "MedMCQA",
    "MedQA - US - 4 options - test": "MedQA",
    "PubMedQA - PQA-L with ground truth labels": "PubMedQA"
}
result["Dataset"] = result["Dataset"].map(dataset_mapping)

# Print the result
print("\nPercentage of Near-Exact Matches by Dataset:")
print(result)

# Create the first bar chart with all 9 datasets
plt.figure(figsize=(12, 6))
plt.bar(result["Dataset"], result["Percentage of Near-Exact Matches"])
plt.title("Percentage of Near-Exact Matches for All Datasets")
plt.xlabel("Dataset")
plt.ylabel("Percentage of Near-Exact Matches")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("all_datasets_chart.png")
plt.close()

# Create the second bar chart with MMLU datasets combined
mmlu_datasets = [
    "MMLU - anatomy",
    "MMLU - clinical knowledge",
    "MMLU - college biology",
    "MMLU - college medicine",
    "MMLU - medical genetics",
    "MMLU - professional medicine"
]

combined_result = result.copy()
mmlu_combined = combined_result[combined_result["Dataset"].isin(mmlu_datasets)]["Percentage of Near-Exact Matches"].mean()
combined_result = combined_result[~combined_result["Dataset"].isin(mmlu_datasets)]
combined_result = pd.concat([combined_result, pd.DataFrame({
    "Dataset": ["MMLU - clinical topics"],
    "Percentage of Near-Exact Matches": [mmlu_combined]
})], ignore_index=True)

plt.figure(figsize=(12, 6))
bars = plt.bar(combined_result["Dataset"], combined_result["Percentage of Near-Exact Matches"])
plt.title("Percentage of Near-Exact Matches (MMLU Combined)")
plt.xlabel("Dataset")
plt.ylabel("Percentage of Near-Exact Matches")
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 20)  # Set y-axis limit to 20%

# Add percentage labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}%',
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig("combined_datasets_chart.png")
plt.close()

print("\nCharts have been saved as 'all_datasets_chart.png' and 'combined_datasets_chart.png'")

def plot_threshold_vs_matches(df):
    thresholds = np.arange(0, 1.01, 0.01)
    total_samples = len(df)
    percentages = np.array([sum(df['Levenshtein Ratio'] >= t) / total_samples * 100 for t in thresholds])

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot all points in blue
    ax.plot(thresholds, percentages, marker='o', color='blue', linestyle='-', markersize=4)

    # Highlight points with threshold > 0.9 in red
    high_threshold_mask = thresholds > 0.9
    ax.plot(thresholds[high_threshold_mask], percentages[high_threshold_mask],
            marker='o', color='red', linestyle='', markersize=6)

    ax.set_title("Percentage of Near-Exact Matches vs Threshold")
    ax.set_xlabel("Near-Exact Threshold")
    ax.set_ylabel("Percentage of Near-Exact Matches")
    ax.grid(True)

    # Add text annotation for the 90% threshold
    ax.axvline(x=0.9, color='gray', linestyle='--')
    ax.text(0.87, 0.95, '90% threshold', transform=ax.transAxes, verticalalignment='top')

    # Add y-values for x=0.9 and x=1.0
    y_90 = percentages[np.argmin(np.abs(thresholds - 0.9))]
    y_100 = percentages[-1]
    ax.text(0.95, y_90 + 2, f'{y_90:.2f}%', verticalalignment='bottom', horizontalalignment='right')
    ax.text(1.042, y_100 + 2, f'{y_100:.2f}%', verticalalignment='bottom', horizontalalignment='right')

    plt.tight_layout()
    plt.savefig("threshold_vs_percentage_matches.png", dpi=300)
    plt.close()

    print("\nThreshold vs Percentage Matches chart has been saved as 'threshold_vs_percentage_matches.png'")

# Call the new function after creating the other charts
plot_threshold_vs_matches(df_cleaned)