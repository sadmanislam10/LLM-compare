import os
import time 
import statistics
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"),
        http_options=types.HttpOptions(timeout=60000),
        )

MODELS = ["gemini-3.5-flash", "gemma-4-26b-a4b-it", "gemini-3.1-flash-lite"]
PROMPTS = [
    {"category": "easy", "prompt": "Explain what an API is in two sentences."},
    {"category": "math", "prompt": "what is 17*23? Give only the final answer."},
    {"category": "coding", "prompt": "Write a Python function that checks if a string is a palindrome. Code only."},
    {"category": "reasoning", "prompt": "A farmer has 17 sheep. All but 9 run away. How many are left? Give only the number."},
]

REPEATS = 3

def ask(model,prompt):
    start = time.perf_counter()
    response = client.models.generate_content(model=model, contents=prompt)
    seconds = time.perf_counter() - start
    usage = response.usage_metadata

    return {
        "model" : model,
        "text" : response.text,
        "seconds" : seconds,
        "input" : usage.prompt_token_count,
        "output" : usage.candidates_token_count,
        "thinking": usage.thoughts_token_count or 0,
        "total": usage.total_token_count,
    }

totals = {}
for model in MODELS:
    totals[model] = {"times": [], "tokens": [], "failed": 0}

for item in PROMPTS:
    print("=" * 60)
    print(f"[{item['category']}] {item['prompt']}")
    for model in MODELS:
        for run in range(REPEATS):
            try:
                result = ask(model, item["prompt"])
            except Exception as error:
                print(f"    {model} (run {run + 1}): FAILED - {error}")
                totals[model]["failed"] += 1
                continue
            totals[model]["times"].append(result["seconds"])
            totals[model]["tokens"].append(result["total"])
            print(f"    {model} (run {run + 1}): "
                  f"{result['seconds']:.1f}s, tokens={result['total']}")

print("=" * 60)
print("SUMMARY")
for model in MODELS:
    t = totals[model]
    if not t["times"]:
        print(f"{model}: all requests failed")
        continue
    print(f"{model}: ok={len(t['times'])}, failed={t['failed']}, "
          f"median time={statistics.median(t['times']):.1f}s, "
          f"median tokens={statistics.median(t['tokens']):.0f}")