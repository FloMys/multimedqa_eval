import json
import tiktoken
from collections import defaultdict

def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens

def sanity_check_jsonl(file_path):
    format_errors = defaultdict(int)
    n_missing_system = 0
    n_missing_user = 0
    n_messages = []
    convo_lens = []
    assistant_message_lens = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                format_errors["json_decode_error"] += 1
                continue

            if not isinstance(example, dict):
                format_errors["data_type"] += 1
                continue

            messages = example.get("messages", None)
            if not messages:
                format_errors["missing_messages_list"] += 1
                continue

            if not any(message.get("role") == "system" for message in messages):
                n_missing_system += 1

            if not any(message.get("role") == "user" for message in messages):
                n_missing_user += 1

            n_messages.append(len(messages))
            convo_lens.append(num_tokens_from_messages(messages))

            for message in messages:
                if "role" not in message or "content" not in message:
                    format_errors["message_missing_key"] += 1

                if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
                    format_errors["message_unrecognized_key"] += 1

                if message.get("role") not in ("system", "user", "assistant", "function"):
                    format_errors["unrecognized_role"] += 1

                content = message.get("content", None)
                function_call = message.get("function_call", None)

                if (not content and not function_call) or not isinstance(content, str):
                    format_errors["missing_content"] += 1

                if message.get("role") == "assistant":
                    assistant_message_lens.append(len(tiktoken.get_encoding("cl100k_base").encode(message.get("content", ""))))

            if not any(message.get("role") == "assistant" for message in messages):
                format_errors["example_missing_assistant_message"] += 1

    print("Format Errors:")
    for error_type, count in format_errors.items():
        print(f"{error_type}: {count}")

    print(f"\nExamples missing system message: {n_missing_system}")
    print(f"Examples missing user message: {n_missing_user}")

    print(f"\nNumber of messages per example:")
    print(f"Min: {min(n_messages)}, Max: {max(n_messages)}")
    print(f"Mean: {sum(n_messages) / len(n_messages):.2f}")

    print(f"\nTotal tokens per example:")
    print(f"Min: {min(convo_lens)}, Max: {max(convo_lens)}")
    print(f"Mean: {sum(convo_lens) / len(convo_lens):.2f}")

    print(f"\nAssistant message tokens per example:")
    print(f"Min: {min(assistant_message_lens)}, Max: {max(assistant_message_lens)}")
    print(f"Mean: {sum(assistant_message_lens) / len(assistant_message_lens):.2f}")

    n_too_long = sum(l > 16385 for l in convo_lens)
    print(f"\n{n_too_long} examples exceed the 16,385 token limit and will be truncated")

    total_tokens = sum(convo_lens)
    print(f"\nTotal tokens in dataset: {total_tokens}")
    print(f"Estimated cost for fine-tuning: ${total_tokens * 0.0080 / 1000:.2f} (at $0.0080 / 1K tokens)")

if __name__ == "__main__":
    jsonl_file_path = "/Users/fmys/Documents/Data/master_thesis/finetuning_data/MultiMedQA_finetuning.jsonl"
    sanity_check_jsonl(jsonl_file_path)