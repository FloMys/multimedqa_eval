"""
Running this script costs about 125 USD!!! 85 USD for GPT4 turbo and 40 USD for GPT4o!
"""
import os
from openai import OpenAI
from datetime import datetime

client = OpenAI(api_key="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")

# TODO: add file path to "batches_to_upload" dir
# Function to save batch info to a file
def save_batch_info(batch_obj, prompt_type, model_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"batch_info_{prompt_type}_{model_name}_{timestamp}.txt"

    # Create the directory if it doesn't exist
    os.makedirs("uploaded_batch_ids", exist_ok=True)

    filepath = os.path.join("uploaded_batch_ids", filename)

    with open(filepath, 'w') as f:
        f.write(f"Batch ID: {batch_obj.id}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Prompt Type: {prompt_type}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Full Batch Object:\n{batch_obj}\n")

    print(f"Batch info saved to {filepath}")


# List of JSONL files
jsonl_files = [
    "MultiMedQA_0shot_GPT4turbo.jsonl",
    "MultiMedQA_5shot_GPT4turbo.jsonl",
    "MultiMedQA_0shot_GPT4o.jsonl",
    "MultiMedQA_5shot_GPT4o.jsonl",
    "MultiMedQA_calibration_0shot_GPT4o.jsonl"
]

for file in jsonl_files:
    print(f"Processing {file}...")

    # Extract prompt type and model name from filename
    parts = file.split('_')
    prompt_type = parts[1]
    model_name = parts[2].split('.')[0]

    # Upload file
    batch_input_file = client.files.create(
        file=open(file, "rb"),
        purpose="batch"
    )

    print(f"Batch Input File ID: {batch_input_file.id}")

    # Create batch
    batch_obj = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"MultiMedQA eval job - {prompt_type} {model_name}"
        }
    )

    print(f"Batch Created Successfully for {file}!")

    # Save batch info
    save_batch_info(batch_obj, prompt_type, model_name)

    print(f"Finished processing {file}\n")
