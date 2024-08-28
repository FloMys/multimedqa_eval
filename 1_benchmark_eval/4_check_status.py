import json
from openai import OpenAI
from pprint import pprint

# Initialize the OpenAI client
client = OpenAI(api_key="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")

# Retrieve the list of 4 most recent batches
batches_list = client.batches.list(limit=4)  # most recent 4 batches

# Pretty print each batch
for i, batch in enumerate(batches_list.data, 4):
    print(f"\nBatch {i}:")
    pprint(json.loads(batch.model_dump_json()))
    print("-" * 50)