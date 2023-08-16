from llama_index import GPTVectorStoreIndex
import os
import gradio as gr

os.environ['OPEN_API_KEY'] = "<API-KEY>"

def chatbot(input_text):
    model = GPTVectorStoreIndex.load_from_disk('model.json')
    answer = model.query(input_text, response_mode="compact")
    return answer.response

app = gr.Interface(fn=chatbot,
                   inputs=gr.inputs.Textbox(lines=5,label="Send a message"),
                   outputs="text",
                   title="Tekhmos Chatbot")

app.launch(share=False)