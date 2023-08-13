# Chatbot background
## NLP (Natural Language Processing)
Example: We want to train a model in order to make it classify documents in a positive box comment or in negative box comment, the first thing to come up in our mind is using a normal Neural Network like this one, the input will be the text, then the hidden layers decides in which output perceptron of the output layer is the text classified (positive or negative).

![NLP](https://github.com/RogerCL24/ChatBot/assets/90930371/bf26611d-1397-4c58-ae59-16e19dadcf2d)

**Problem**

We cannot hand over text inputs to the perceptrons, we already know the perceptron does a summaton of the inputs multiplied by the weights, we can not multiply _"hello" * 0.345 = ?_, therefore we will not be able to propagate forward the data and that means we can not calculate the error and update the weights to auto-improve the model.

**Solution**

- Text to number conversion, there are several techniques:
  - _Random numbers_: Allocate a random number to each character or word. The ASCII code.
    
    ![Standard-ASCII-Table_large](https://github.com/RogerCL24/ChatBot/assets/90930371/dfaa0c19-9ac6-41f4-ad61-dfe1ded97240)
    
    Using this technique makes the model or the neural network focus on the character too much (redundantly detailed), that produces the model to lose focus on the context. Therefore we have to do a zoom out and aim the words, namely, we number the words instead of the characters.

  For instance, `Codificar` = 1 & `Palabra` = 2, then we use a matrix N x N, where N is equal to the total number of words in our vocabulary. <sub> In this case N = 2 </sub>
  
  ![Matrix](https://github.com/RogerCL24/ChatBot/assets/90930371/3dd41e2c-2bc9-4019-b0ca-d9fa1766d6b0)

  Wherever there is a match between the row and the column a 1 will be placed <sub> only can be one 1 in each row and column </sub>, otherwise we place a 0. The process to turn words into numbers or vectors (matrix) is called ``tokenization``.
  A token is each word or subword numerically coded
 
  - _One - Hot - Encoding_: Similar to _random_numbers_ technique, we have a N x N square matrix, where N is the total of words of our vocabulary.

  ![One hot](https://github.com/RogerCL24/ChatBot/assets/90930371/acaac097-a2e3-4952-8353-22b53ee90643)

  Each vector is a word, now the word 'Codificar' is represented as the 1,0,0,0 vector, therefore when the neural network reads a 1,0,0,0 vector knows that is the word 'Codificar', each vector is a different token.

  - _Embeddings_: As everyone knows matrix has a huge computacional cost in the our computer performance, so let's going to reduce the matrix size joinning the NLP (one-hot-encoding) with neural networks.
   
  First we have this matrix 4 x 4.
  
  ![MAt](https://github.com/RogerCL24/ChatBot/assets/90930371/4260d548-b3de-4d72-8a13-c6332bff2e61)

  Second, in order to compact the matrix to make our NLP more efficiente we give the one-hot-encoded word (the vector) as an input data to the Neural Network architecture, the embedding.
  
  ![NN](https://github.com/RogerCL24/ChatBot/assets/90930371/dd8d0eff-8f8f-41a3-a670-e42b683443f0)

  Finally, the output layer gives us another vector with the size we want, for instance 4 x 1.
  
  ![Resot](https://github.com/RogerCL24/ChatBot/assets/90930371/4a5045fa-4762-494b-91b5-81be06d116cb)

  A simple real sample would be this one, where each word is the data input (cat, kitten...) and the columns are the context (feline, human...).
  
  ![Embedding](https://github.com/RogerCL24/ChatBot/assets/90930371/23984e52-9827-4219-a273-31cbc74453df)

  As we are humans we can not represent 7D objects, consequently we use dimensionality reduction techniques to give a proper represention of the embedded words, as you can see, for instance, cat and kitten are more close than cat and houses.
  This way we are allowed to make vector operations, for instance if I subtract queen to king it should give women

## RNN

## Transformers
