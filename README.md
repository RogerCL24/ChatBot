# Chatbot background
> This repository has been made after [ANN](https://github.com/RogerCL24/ANN), some concepts are explained there.
## NLP (Natural Language Processing)
Example: We want to train a model in order to make it classify documents in a positive box comment or in negative box comment, the first thing to come up in our mind is using a normal Neural Network like this one, the input will be the text, then the hidden layers decides in which output perceptron of the output layer is the text classified (positive or negative).

![NLP](https://github.com/RogerCL24/ChatBot/assets/90930371/bf26611d-1397-4c58-ae59-16e19dadcf2d)

### Problem

We cannot hand over text inputs to the perceptrons, we already know the perceptron does a summaton of the inputs multiplied by the weights, we can not multiply _"hello" * 0.345 = ?_, therefore we will not be able to propagate forward the data and that means we can not calculate the error and update the weights to auto-improve the model.

### Solution

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

  Second, in order to compact the matrix to make our NLP more efficient we give the one-hot-encoded word (the vector) as an input data to the Neural Network architecture, the embedding.
  
  ![NN](https://github.com/RogerCL24/ChatBot/assets/90930371/dd8d0eff-8f8f-41a3-a670-e42b683443f0)

  Finally, the output layer gives us another vector with the size we want, for instance 4 x 1.
  
  ![Resot](https://github.com/RogerCL24/ChatBot/assets/90930371/4a5045fa-4762-494b-91b5-81be06d116cb)

  A simple real sample would be this one, where each word is the data input (cat, kitten...) and the columns are the context (feline, human...).
  
  ![Embedding](https://github.com/RogerCL24/ChatBot/assets/90930371/23984e52-9827-4219-a273-31cbc74453df)

  As we are humans we can not represent 7D objects, consequently we use dimensionality reduction techniques to give a proper represention of the embedded words. As you can see, for instance, cat and kitten are more close than cat and houses.
  This way we are allowed to make vector operations, for instance if I subtract queen to king it should give women.

## RNN (Recurrent Neural Metwork)
Firsty, what is 'memory', memory is a brain function that allows the organism code, store and retrieve ``past`` information. Up to now we have been using a neural network that classifies images that do not dependend of any time var, if the image is a dog our model will classify it as a dog (if it's well implemented) regardless any other input.

### Problem
The problem comes when we have sequences, where the sequence elements has a logic order. For intance, a video is a sequence of images indexed in time or a text, a sequence of words with a logic order to give sense to the word sequence. 
So the video or the text can not be input data of this type of neural networks, because each neuron only can store 1 data at the same time, and the sequences depends temporarily from the previous items of the sequence, so we need more than 1 input here.

### Solution
We change the neural network architecture to RNN, now we have another input at the same neuron to modulate the output secuence order, depending indeed on the time `t = 1, t = 2, t = 3,...,t = n`

![RNN](https://github.com/RogerCL24/ChatBot/assets/90930371/ca21162b-46d2-4be2-8893-cb512256b512)

### Explanation
![neuron](https://github.com/RogerCL24/ChatBot/assets/90930371/670868b2-e5cf-4153-8874-41cc770aaf55)

Focus on the left neuron, it has an input `X`, a summation `∑`, the activation function and an output `ŷ` up to now it is as always, but now we got another output that feeds back our neuron making it a new input.

Now the right figure which has several neurons, it has not, is the same neuron through the time, as you can observe we have the first input `X(0)` -> _'La'_: so `ŷ(0)` will be _'The'_, ``X(1)`` -> _'manzana'_: `ŷ(1)` will be _'The apple'_ because we have the other input data from before, besides the current one, `X(2)` -> _'roja'_: so `ŷ(2)` will be _'The red apple'_, as you see the order changes due to the grammatical rules, for that reason is important this architecture to follow the sequence, finally `X(3)` -> _'vuela'_: so `ŷ(3)` will be _'The red apple flies'_, cause the feed back input returns _'The red apple'_.

That was only 1 neuron, in a layer would be like this.

![lay](https://github.com/RogerCL24/ChatBot/assets/90930371/076f2dc2-5c8d-42fe-bb0e-450e728debe8)

Same functionality but with more neurons to improve performance, obviously the data used as feed back is stored in memory in order to keep it for further inputs, that produces a problem of memory with long sequences, there too many words to store and we can not save them all, **solution** -> Transformers ⬇️

## Transformers

As a means to solve the former problem of memory loss, Google published the paper _Attention all you need_ at 2017, this paper basically put forward the new model architecture - **The transformer** - .

<p align="center">
  
<img src="https://github.com/RogerCL24/ChatBot/assets/90930371/3b441e25-3b3e-4ad1-896d-340703af74f4"/>

</p>






