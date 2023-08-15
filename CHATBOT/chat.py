import openai

openai.api_key = "<API-KEY>"

while True:

    prompt=input("\n Send a message: ")
    
    if prompt == 'Leave' or prompt == 'leave':
        break

    chatbot = openai.Completion.create(engine="text-davinci-003",
                                    prompt=prompt,
                                    max_tokens=2048)

    print(chatbot.choices[0].text)