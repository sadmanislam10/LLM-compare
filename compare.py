import os
import time
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
    totals[model] = {"seconds": 0, "tokens": 0, "ok": 0, "failed": 0}
    
for item in PROMPTS:
    print("=" * 60)
    print(f"[{item['category']}] {item['prompt']}")
    for model in MODELS:
        try:
            result = ask(model,item["prompt"])
        except Exception as error:
            print(f"    {model}: FAILED - {error}")
            totals[model]["failed"] += 1
            continue
        totals[model]["ok"] += 1
        totals[model]["seconds"] += result["seconds"]
        totals[model]["tokens"] += result["total"]
        preview = result["text"].replace("\n", " ")[:100]
        print(f"    {model}: {result['seconds']:.1f}s, total tokens={result['total']}")
        print(f"    {preview}")
        
print("=" * 60)
print("SUMMARY")
for model in MODELS:
    t = totals[model]
    avg = t["seconds"] / t["ok"] if t["ok"] else 0
    print(f"{model}: ok={t['ok']}, failed={t['failed']}, "
      f"time={t['seconds']:.1f}s, avg={avg:.1f}s, tokens={t['tokens']}")