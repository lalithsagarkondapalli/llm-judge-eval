"""OpenAI / Azure OpenAI client factory with retries."""
import json
import os
import time

CRITERIA = ["factual_accuracy", "groundedness", "completeness", "relevance"]


def get_client():
    from openai import AzureOpenAI, OpenAI

    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        return AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )
    return OpenAI()  # reads OPENAI_API_KEY


def chat(client, model, messages, temperature=0.0, response_format=None, max_retries=5):
    kwargs = dict(model=model, messages=messages, temperature=temperature)
    if response_format:
        kwargs["response_format"] = response_format
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(**kwargs)
            return r.choices[0].message.content, r.usage
        except Exception as e:  # rate limits, transient errors
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  retry in {wait}s ({type(e).__name__}: {e})")
            time.sleep(wait)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
