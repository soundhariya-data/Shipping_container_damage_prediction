from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model=" gemini-1.5-flash",

    google_api_key="AIzaSyCU6pBqGqNUNtMIq2yubzqDVu4OgtjrnyE"
)

response = llm.invoke("Say hello in one sentence")
print(response.content)

