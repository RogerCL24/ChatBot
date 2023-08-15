# CHATBOT 

## DOCUMENTATION

### OpenAI API
We visit the [OpenAI](https://platform.openai.com/docs/api-reference) web page, then we log and go to Manage Account > Usage, we will see that we have $18.00 for a free trial usage:

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

## SETUP

We are going to be using the _Jupyter_ notebooks at VScode to implement the chatbot.

### Virtual environment
In this case we selected venv of python to create the virtual environment, you can use anaconda in your cmd as well, but commands will differ.

- Create the workpace and the directory which you are going to create the virtual environment, then use this prompt in VScode cmd to create the venv:

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

Finally we create a `.ipynb` file in the same directory as the venv, <sub> You need to install _Jupyter_ extension from microsoft if you do not have it already </sub>, then we select the virtual environment where the file is going to be executed at `Select kernel` <sub> Upper right corner </sub>, we will select _Python Environments_ -> and see the venv named **kernel_name** <sub> from the former code </sub>
