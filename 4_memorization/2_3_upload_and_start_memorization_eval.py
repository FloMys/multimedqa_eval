"""
Running this script costs about 3.30 USD
"""
import os
from openai import OpenAI
from datetime import datetime


client = OpenAI(api_key="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")


def save_batch_info(batch_obj, file):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"batch_info_{file}.txt"

    # Create the directory if it doesn't exist
    os.makedirs("uploaded_batch_ids", exist_ok=True)

    filepath = os.path.join("uploaded_batch_ids", filename)

    with open(filepath, 'w') as f:
        f.write(f"Batch ID: {batch_obj.id}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Full Batch Object:\n{batch_obj}\n")

    print(f"Batch info saved to {filepath}")


# List of JSONL files
jsonl_files = [
    "MultiMedQA_memorization_GPT4o.jsonl"
]

for file in jsonl_files:
    print(f"Processing {file}...")

    # Upload file
    batch_input_file = client.files.create(
        file=open("batches_to_upload/" + file, "rb"),
        purpose="batch"
    )

    print(f"Batch Input File ID: {batch_input_file.id}")

    # Create batch
    batch_obj = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": file
        }
    )

    print(f"Batch Created Successfully for {file}!")

    # Save batch info
    save_batch_info(batch_obj, file)

    print(f"Finished processing {file}\n")
