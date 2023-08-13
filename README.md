# Chatbot background
## NLP (Natural Language Processing)
Example: We want to train a model in order to make it classify documents in a positive box comment or in negative box comment, the first thing to come up in our mind is using a normal Neural Network like this one, the input will be the text, then the hidden layers decides in which output perceptron of the output layer is the text classified (positive or negative).

![NLP](https://github.com/RogerCL24/ChatBot/assets/90930371/bf26611d-1397-4c58-ae59-16e19dadcf2d)

**Problem**

We cannot hand over text inputs to the perceptrons, we already know the perceptron does a summaton of the inputs multiplied by the weights, we can not multiply _"hello" * 0.345 = ?_, therefore we will not be able to propagate forward the data and that means we can not calculate the error and update the weights to auto-improve the model.

**Solution**

- Text to number conversion, there are several techniques:
  - _Random numbers_: Allocate a random number to each character or word.
    ![Standard-ASCII-Table_large](https://github.com/RogerCL24/ChatBot/assets/90930371/dfaa0c19-9ac6-41f4-ad61-dfe1ded97240)

  - _One - Hot - Encoding_:
  - _Embeddings_:


## RNN

## Transformers
