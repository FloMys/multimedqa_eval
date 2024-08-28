from openai import OpenAI
import os
import re

# Set up OpenAI API credentials
client = OpenAI(api_key="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")

# Directory paths
input_dir = "uploaded_batch_ids"
output_dir = "downloaded_batch_outputs"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read the list of txt files from the provided file
txt_files = ["batch_info_calibration_0shot_GPT4o_20240710_134040.txt"]

for txt_file in txt_files:
    txt_file_path = os.path.join(input_dir, txt_file)

    # Check if the file exists
    if not os.path.exists(txt_file_path):
        print(f"File not found: {txt_file_path}")
        continue

    # Read the batch ID from the file
    with open(txt_file_path, 'r') as f:
        content = f.read()
        batch_id_match = re.search(r'Batch ID: (batch_\w+)', content)
        if not batch_id_match:
            print(f"Batch ID not found in {txt_file}")
            continue
        batch_id = batch_id_match.group(1)

    # Extract information for the output filename
    prompt_type, model, timestamp = re.match(r'batch_info_(\w+)_(\w+)_(\d+_\d+)\.txt', txt_file).groups()

    try:
        # Retrieve the batch information
        batch = client.batches.retrieve(batch_id)
        print(f"\nProcessing batch: {batch_id}")

        # Check if the batch is complete
        if batch.status == "completed":
            # Retrieve and save the output file
            if batch.output_file_id:
                output_content = client.files.content(batch.output_file_id)
                output_filename = f"calibration_eval_{prompt_type}_{model}_{timestamp}_output.jsonl"
                output_path = os.path.join(output_dir, output_filename)
                with open(output_path, "wb") as output_file:
                    output_file.write(output_content.content)
                print(f"Output file saved: {output_path}")

            # Retrieve and save the error file if it exists
            if batch.error_file_id:
                error_content = client.files.content(batch.error_file_id)
                error_filename = f"{prompt_type}_{model}_{timestamp}_errors.jsonl"
                error_path = os.path.join(output_dir, error_filename)
                with open(error_path, "wb") as error_file:
                    error_file.write(error_content.content)
                print(f"Error file saved: {error_path}")
            else:
                print(f"No errors found for batch {batch_id}")
        else:
            print(f"Batch {batch_id} is not complete yet. Status: {batch.status}")

    except Exception as e:
        print(f"Error processing batch {batch_id}: {str(e)}")

print("\nProcessing complete.")