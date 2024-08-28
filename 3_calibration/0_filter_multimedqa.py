import csv
import os
from collections import Counter

# Input and output file paths
input_file = "/Users/fmys/Documents/Data/master_thesis/0_MultiMedQA_final/0_MultiMedQA.csv"
output_file = "/Users/fmys/Documents/Data/master_thesis/0_MultiMedQA_final/0_MultiMedQA_filtered.csv"

# Rows to exclude
exclude_tags = ['MedQA - Mainland - test', 'MedQA - Taiwan - test', 'MedQA - US - test']

# Initialize counters
original_counter = Counter()
filtered_counter = Counter()

# Read the input CSV and write to the output CSV, excluding specified rows
with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
        open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Write the header row
    header = next(reader)
    writer.writerow(header)

    # Find the index of the 'Dataset' column
    dataset_index = header.index('Dataset')

    # Process each row
    for row in reader:
        dataset_tag = row[dataset_index]
        original_counter[dataset_tag] += 1

        if dataset_tag not in exclude_tags:
            writer.writerow(row)
            filtered_counter[dataset_tag] += 1

print(f"Filtered CSV has been saved to: {output_file}")

# Print the counts for each dataset
print("\nOriginal dataset counts:")
print(original_counter)

print("\nFiltered dataset counts:")
print(filtered_counter)

# Calculate total counts
total_original = sum(original_counter.values())
total_filtered = sum(filtered_counter.values())

print(f"\nTotal rows in original file: {total_original}")
print(f"Total rows in filtered file: {total_filtered}")
print(f"Total rows removed: {total_original - total_filtered}")