import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GEMINI_API_KEY")

if key is None:
    print("Key NOT found. Check your .env file.")
else:
    print("Key found!")
    print("Length:", len(key))
    print("Has stray quotes or spaces:", key != key.strip().strip('"').strip("'"))