from google import genai

client = genai.Client(api_key="AIzaSyBhhf5oBVqLEMwzVZ3U96wKoa7Sy0OtMAU")

print("Sending message...")
response = client.models.generate_content(
    model="gemini-2.5-flash",  # ← smallest free model
    contents="Say hello in one sentence"
)

print("Response:")
print(response.text)
