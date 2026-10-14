import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key = os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents="Explain what an API is in two sentences.",
)

print(response.text)

usage = response.usage_metadata

print("Input tokens: ", response.usage_metadata.prompt_token_count)
print("Output tokens: ", response.usage_metadata.candidates_token_count)
print("Thinking tokens:", usage.thoughts_token_count)
print("Total tokens: ",usage.total_token_count)