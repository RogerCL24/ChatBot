import gradio as gr

def greeting(name,daylight,farenheit_temp):
    greeting = "Good morning" if daylight == True else "Good night"
    temperature = round((farenheit_temp-32)*5/9)
    greetings = f"{greeting} {name}. Today's temperature is {temperature} Celsius degrees"
    return greetings
    

app = gr.Interface(fn=greeting,
                   inputs=["text","checkbox", gr.Slider(0,120)],
                   outputs="text")

app.launch()