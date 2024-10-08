import json
from openai import OpenAI
from pprint import pprint

# Initialize the OpenAI client
client = OpenAI(api_key="your-api-key-here")

# Retrieve most recent batch
batches_list = client.batches.list(limit=1)

# Pretty print each batch
for i, batch in enumerate(batches_list.data, 1):
    print(f"\nBatch {i}:")
    pprint(json.loads(batch.model_dump_json()))
    print("-" * 50)