import os
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv(override=True)

c = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ["DEEPSEEK_BASE_URL"],
)
for m in c.models.list().data:
    print(m.id)