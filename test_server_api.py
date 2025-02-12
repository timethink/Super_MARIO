import openai

client = openai.Client(base_url="http://127.0.0.1:30020/v1", api_key="None")

response = client.chat.completions.create(
    model="/workspace/AlphaMath-7B",
    messages=[
        {"role": "user", "content": "List 3 countries and their capitals."},
    ],
    temperature=0,
    max_tokens=64,
)

print(f"Response: {response}")