# CHATBOT (deprecated)

## DOCUMENTATION 📍

### OpenAI API
We visit the [OpenAI](https://platform.openai.com/docs/api-reference) web page, then we log and go to Manage Account > Usage, we will see that we have _$18.00_ <sub> If you sign up before 2023 </sub> or _$5.00_ <sub> Otherwise </sub> for a free trial usage: 

<sub> Otherwise, if you desire a billing to acquire more models and services is available as well</sub>

<p align="center"> 

<img src="https://github.com/RogerCL24/ChatBot/assets/90930371/0fcfbf58-6724-40b1-8ae5-ddbf1a7ec311"/>
</p>

If you want to keep paying in order to use more the api you can go to [https://openai.com/pricing](https://openai.com/pricing) and check the fees. For example, the most advanced up to now, GPT-4 -> 
- Depending on the context amount we have different fees, in the 8K context model they will charge you _$0.03_ for each 1K tokens of prompt/input and _$0.06_ for each 1K tokens of answer/ouput. <sub> Tokens are explained at [Chatbot background](https://github.com/RogerCL24/ChatBot)

<p align="center">

<img src="https://github.com/RogerCL24/ChatBot/assets/90930371/69247689-4c32-4e3d-85ed-90feaedd1751"/>
</p>

In this chatbot we are going to use the `Fine-tuning models`, they are models already created and we only have to train them with our own data, therefore we are fine-tuning their base model.
- There are 4 models, the more expensive the more performance it has.

<p align="center">

<img src="https://github.com/RogerCL24/ChatBot/assets/90930371/4a58611e-eb11-40cd-96ed-e181c8ff9838"/>
</p>

### LangChain

With this framework we can integrate a library in charge to use the OpenAI API and connect it to the following library.⬇️ <sub> More info [https://python.langchain.com/docs/get_started/introduction.html](https://python.langchain.com/docs/get_started/introduction.html) </sub>

### GPT index

With this library we can supply new data to _LangChain_ in order to transform that data into the embeddings needed that will be consumed by the OpenAI API. <sub> More info [https://gpt-index.readthedocs.io/en/stable/](https://gpt-index.readthedocs.io/en/stable/)</sub>

### Gradio

Finally, _Gradio_ can generate interfaces to interact with trained models with a few code lines, dependending on the complexity of the interface that we want. <sub> More info [https://www.gradio.app/guides/quickstart](https://www.gradio.app/guides/quickstart) </sub>

## SETUP 📍

We are going to be using the _Jupyter_ notebooks at VScode to implement the chatbot.

### Virtual environment
In this case we selected venv of python to create the virtual environment, you can use anaconda in your cmd as well, but the following commands will differ.

- Create the workspace and the directory which you are going to create the virtual environment, then use this prompt in VScode cmd to create the venv:

```bash
python3 -m venv virtual_env_name
```
- Now, to activate it:
```bash
source virtual_env_name/bin/activate
```
Fine, now we go with the libraries:

```bash
pip install openai
pip install gradio
pip install gpt-index
pip install langchain
```
And also we are going to need another library in our virtual environment if we want to use this venv as a kernel where our files are going to be executed over.

```bash
pip install ipykernel
python -m ipykernel install --user --name=kernel_name
```

Finally we create a `.ipynb` file in the same directory as the venv, <sub> You need to install _Jupyter_ extension from microsoft if you do not have it already </sub>, then we select the virtual environment where the file is going to be executed at `Select kernel` <sub> Upper right corner </sub>, we will select _Python Environments_ -> and see the venv named **kernel_name** <sub> from the former code </sub>.

## API requests 📍
In order to try the API's requests we have to create an API key first, therefore we log at the OpenAI web page, then here [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys) and finally we create a new key which we have to copy and paste it at `<API-KEY>` gap.

```python
import openai

openai.api_key = "<API-KEY>"

chatbot = openai.Completion.create(engine="text-davinci-003",
                                   prompt="What is ChatGPT?",
                                   max_tokens=2048)
print(chatbot.choices[0].text)
```
Basically we are creating an instance, `chatbot` var. Now we use the ``openai`` library -> then we use the task `Completion` (output) and `create` -> and finally we specify an engine `text-davinci-003` <sub> Check all models [here](https://platform.openai.com/account/rate-limits) </sub>, the prompt (input) `What is ChatGPT?` and the limit of tokens (words) of the answer `2048`. At last we print it, you can see the answer of this code execution at [chat.ipynb](https://github.com/RogerCL24/ChatBot/blob/main/CHATBOT/chat.ipynb).

Now let's do the same but through the console and with any prompt to make sure it works.
```python
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
```
Same as before but with an infinite loop and an input var.
- Example:
<p align="center">

<img src="https://github.com/RogerCL24/ChatBot/assets/90930371/52052f76-57c6-4797-abed-f5c80b992c3c"/>
</p>

<a name="frontend-with-gradio"></a>
## Frontend with Gradio 📍
Gradio will allow us to create an easy coded web page where we can use our chatbot, but first let's see how it works with a little example: 
```python
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
```
We define a var as `app` which is an interface (``gradio`` library and ``Interface`` object), `Interface` has 3 parameters:
- **Function (fn)**: In this case `greeting`, which has at the same time 3 parameters and returns a text.
- **inputs**: The inputs for the `greeting` function.
- **outputs**: The ouput from the `greeting` function.

When we execute this code at the console it will print a link, we click it `Ctrl+left_click` and then it will redirect us to a local host with the web page formerly implemented:
<p align="center">

<img src="https://github.com/RogerCL24/ChatBot/assets/90930371/c1ffe326-9499-4a96-9528-1cc1a06d14a4"/>
</p>

## Training 📍
> [!NOTE]
> Code from [training.py](training.py)

- First, we store the `API-KEY` at the enviornment var with:
```python
os.environ['OPENAI_API_KEY'] = "<API-KEY>"
```
> [!IMPORTANT]
> ``import os`` library

- Then we define the parameters we are going to use:
```python
max_input = 4098
tokens = 256
chunk_size = 600
max_chunk_overlap = 0.1
```
-
  - _max_input_: Max input/prompt size allowed, namely, max number of characters.
  - _tokens_: Max amount of tokens from the prompt.
  - _chunck_size_: Max prompt chunk (fragment) size allowed after being split, that means, up to how many tokens can each chunk have from the split prompt.
  - _max_chunk_overlap_: Max allowed value for the overlapping between chunks of the prompt. We have to know that the prompts are split in littles pieces (chunks) in order to process them. **Example** -> the words _similar_ and _similarity_, the tokens of the word _similar_ can overlap the tokens of the word _similarity_, as ther root of the word is the same.

- Finally we define the function to train our model `text-ada-001`:
```python

def training(path):
    openai.api_key = os.environ["OPENAI_API_KEY"]
    docs = SimpleDirectoryReader(path).load_data()
    Prompt_helper = PromptHelper(max_input, tokens, max_chunk_overlap, chunk_size_limit=chunk_size)
    model = LLMPredictor(llm=OpenAI(temperature=0,model_name="text-ada-001",max_tokens=tokens))
    context = ServiceContext.from_defaults(llm_predictor=model, prompt_helper=Prompt_helper)

    index_model = GPTVectorStoreIndex.from_documents(docs, service_context=context)
    index_model.storage_context.persist(persist_dir='Store')

training("data")
```
> [!NOTE]
> `data` is the directory where the _.text_ docs are placed, [data](data).
> Besides we need all the libraries from [training.py](training.py)

- 
  - **_openai.api_**: We have to grant our key to openai in order to use their services.
  - **_docs_**: Read the docs placed at `path`.
  - **_Prompt_helper_**: Personalize the prompt features in order to meet our needs and helps the LLM deal with the context, most of the parameters if they are not defined at the argument space can be unlimited <sub> For instance, ``chunk_size_limit`` value would be none </sub>.
  - **_model_**: Which model will be used, we specify from what source is that model, `llm=OpenAI`, and the parameters:
    - **_temperature_**: Cost level we can reach, 0 means free trial usage and the response will be deterministic <sub> Always returns the same completion </sub>, a higher temperature value makes the completion more diverse and creative.
    - **_model_name_**: Mode name of the source (OpenAI in this case).
    - **_max_tokens_**: The maximum number of tokens to generate in the completion.
    
  <sub> More info about the parameters, [https://api.python.langchain.com/en/latest/llms/langchain.llms.openai.OpenAI.html](https://api.python.langchain.com/en/latest/llms/langchain.llms.openai.OpenAI.html) </sub>
  - **_context_**: We define which model we are going to use and the prompt helper that is going to lead that model.
  - **_index_model_**: We take a set of text documents, calculate the vectors for these documents using our model, and then build a vector index[^1] (``JSON`` object) to facilitate efficient search and retrieval of similar documents in the future.
- Finally, we store the JSON object in our disk, the directory where we store it is _Store_ in this case.

## Predict 📍
> [!NOTE]
> Code from [predict.py](predict.py)

- Store the API key as before

### Function
- Define the function which is going to update the interface depending on the `input_text`:
```python
def chatbot(input_text):
    openai.api_key = os.environ['OPENAI_API_KEY']
    storage_context = StorageContext.from_defaults(persist_dir='Store')
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    response = query_engine.query(input_text)
    return response.response
```
- - **_storage_context_**: To load data, we simply need to re-create the storage context using the same configuration (e.g. pass in the same persist_dir). 
  - **_index_**: We can then load specific indices from the StorageContext, in this case we don't need to specify _index_id_ (another parameter) if there's only one index in storage context.
  - **_query_engine_**[^2]: Creates a Retriever instance with the default values for a given index retriever[^3].
  - **_response_**: In order to use the ``query_engine`` we pass the query (_input_text_) as a parameter and store the output (_response_) to stream it later.
    
### Interface
- We will be using `Gradio` to generate the interface like in the former example ➡️ [📍](#frontend-with-gradio):
```python
app = gr.Interface(fn=chatbot,
                   inputs=gr.inputs.Textbox(lines=5,label="Send a message"),
                   outputs="text",
                   title="Tekhmos Chatbot")
app.launch(share=False)
```
> [!NOTE]
> **share** is set to _False_ by default, setting _True_ will generate a public ip during 72 hours to access the chatbot interface generated by `Gradio`.



## Example & Conclusion 📍
This example will be done comparing the output of the trained chatbot <sub> With [data](data) docs </sub> and the [datos2.txt](data/datos2.txt) document using an specific prompt to test if the model has truly learned/trained.
<p align="center">

<img src="https://github.com/RogerCL24/ChatBot/assets/90930371/947cf7a3-e99f-41e8-ac6b-921cd9b16bf6"/>
</p>

<sub> Text from [datos2.txt](data/datos2.txt) document </sub>
<p align="center"> 

<img src="https://github.com/RogerCL24/ChatBot/assets/90930371/fc29fc7a-92d8-473e-b178-b3f2fd7ec9e6"/>
</p>

- The prompt is _Talk me about the education and the AI in the future_, make sense because we trained it with that data, as you can see our model uses sentences and information from the ``datos2.txt`` document.

In conclusion, we can train pre-trained models with the fine-tuning models from _ChatGPT_ with our own data in order to provide further knowledge in specific fields such as recent affairs from the world which the model you are using is not trained with, or give virtual support to institutions like schools, colleges, hospitals, banks....


[^1]: What is an index? Jerry Liu wrote "An index manages the state: abstracting away underlying storage, and exposing a view over processed data & associated metadata.” LlamaIndex indexes nodes. Nodes are chunks of the documents we load from the storage. More info at [https://gpt-index.readthedocs.io/en/latest/core_modules/data_modules/index/root.html](https://gpt-index.readthedocs.io/en/latest/core_modules/data_modules/index/root.html)
[^2]: Takes in a natural language query, and returns a rich response. It is most often (but not always) built on one or many Indices via Retrievers. You can compose multiple query engines to achieve more advanced capability. More info at [https://gpt-index.readthedocs.io/en/latest/core_modules/query_modules/query_engine/root.html](https://gpt-index.readthedocs.io/en/latest/core_modules/query_modules/query_engine/root.html)
[^3]: Retrievers are responsible for fetching the most relevant context given a user query (or chat message). More info at [https://gpt-index.readthedocs.io/en/latest/core_modules/query_modules/retriever/root.html](https://gpt-index.readthedocs.io/en/latest/core_modules/query_modules/retriever/root.html)
