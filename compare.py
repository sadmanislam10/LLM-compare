import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"),
        http_options=types.HttpOptions(timeout=60000),
        )

MODELS = ["gemini-3.5-flash","gemma-4-31b-it"]
PROMPT = "Explain what an API is in two sentences."

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

for model in MODELS:
    print("=" * 50)
    print(model)
    try:
        result = ask(model, PROMPT)
    except Exception as error:
        print("FAILED:", error)
        continue
    print(result["text"])
    print(f"Time: {result['seconds']:.1f}s")
    print(f"Tokens: input={result['input']}, output={result['output']}, "
          f"thinking={result['thinking']}, total={result['total']}")
   