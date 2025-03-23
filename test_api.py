import openai 
import dotenv 
import os 

dotenv.load_dotenv(override=True) 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 

client = openai.OpenAI(api_key=OPENAI_API_KEY) 
response = client.chat.completions.create(
    model = "gpt-4o-mini", 
    messages=[{
        "role": "user", 
        "content": "Send me a summary in return of the great gatsby."
    }], 
)

print(response.choices[0].message.content) 