# Chatbot background
> This repository has been made after [ANN](https://github.com/RogerCL24/ANN), some concepts are already explained there.
## NLP (Natural Language Processing)
Example: We want to train a model in order to make it classify documents in a positive box comment or in negative box comment, the first thing to come up in our mind is using a normal Neural Network like this one, the input will be the text, then the hidden layers decides in which output perceptron of the output layer is the text classified (positive or negative).

<p align="center">

<img src="https://github.com/RogerCL24/ChatBot/assets/90930371/bf26611d-1397-4c58-ae59-16e19dadcf2d"/>
</p>

### Problem

We cannot hand over text inputs to the perceptrons, we already know the perceptron does a summaton of the inputs multiplied by the weights, we can not multiply _"hello" * 0.345 = ?_, therefore we will not be able to propagate forward the data and that means we can not calculate the error and update the weights to auto-improve the model.

### Solution

- Text to number conversion, there are several techniques:
  - _Random numbers_: Allocate a random number to each character or word. The ASCII code.

 <p align="center">
      
 <img src="https://github.com/RogerCL24/ChatBot/assets/90930371/dfaa0c19-9ac6-41f4-ad61-dfe1ded97240"/>
 </p>
    
  Using this technique makes the model or the neural network focus on the character too much (redundantly detailed), that produces the model to lose focus on the context. Therefore we have to do a zoom out and aim the words, namely, we number the words instead of the characters.

  For instance, `Codificar` = 1 & `Palabra` = 2, then we use a matrix N x N, where N is equal to the total number of words in our vocabulary. <sub> In this case N = 2 </sub>

  <p align="center"> 
  
  <img src="https://github.com/RogerCL24/ChatBot/assets/90930371/3dd41e2c-2bc9-4019-b0ca-d9fa1766d6b0"/>
  </p>
  
  Wherever there is a match between the row and the column a 1 will be placed <sub> only can be one 1 in each row and column </sub>, otherwise we place a 0. The process to turn words into numbers or vectors (matrix) is called ``tokenization``.
  A token is each word or subword numerically coded
 
  - _One - Hot - Encoding_: Similar to _random_numbers_ technique, we have a N x N square matrix, where N is the total of words of our vocabulary.

  <p align="center">
    
  <img src="https://github.com/RogerCL24/ChatBot/assets/90930371/acaac097-a2e3-4952-8353-22b53ee90643"/>
  </p>

  Each vector is a word, now the word 'Codificar' is represented as the 1,0,0,0 vector, therefore when the neural network reads a 1,0,0,0 vector knows that is the word 'Codificar', each vector is a different token.

  - _Embeddings_: As everyone knows matrix has a huge computacional cost in the our computer performance, so let's going to reduce the matrix size joinning the NLP (one-hot-encoding) with neural networks.
   
  First we have this matrix 4 x 4.
  
  <p align="center">
  
  <img src="https://github.com/RogerCL24/ChatBot/assets/90930371/4260d548-b3de-4d72-8a13-c6332bff2e61"/>
  </p>

  Second, in order to compact the matrix to make our NLP more efficient we give the one-hot-encoded word (the vector) as an input data to the Neural Network architecture, the embedding.

  <p align="center">
    
  <img src="https://github.com/RogerCL24/ChatBot/assets/90930371/dd8d0eff-8f8f-41a3-a670-e42b683443f0"/>
  </p>
  
  Finally, the output layer gives us another vector with the size we want, for instance 4 x 1.

  <p align="center">
    
  <img src="https://github.com/RogerCL24/ChatBot/assets/90930371/4a5045fa-4762-494b-91b5-81be06d116cb"/>
  </p>

  A simple real sample would be this one, where each word is the data input (cat, kitten...) and the columns are the context (feline, human...).

  <p align="center">
  
  <img src="https://github.com/RogerCL24/ChatBot/assets/90930371/23984e52-9827-4219-a273-31cbc74453df"/>
  </p>

  As we are humans we can not represent 7D objects, consequently we use dimensionality reduction techniques to give a proper represention of the embedded words. As you can see, for instance, cat and kitten are more close than cat and houses.
  This way we are allowed to make vector operations, for instance if I subtract queen to king it should give women.

## RNN (Recurrent Neural Network)
Firsty, what is 'memory', memory is a brain function that allows the organism code, store and retrieve ``past`` information. Up to now we have been using a neural network that classifies images that do not dependend of any time var, if the image is a dog our model will classify it as a dog (if it's well implemented) regardless any other input.

### Problem
The problem comes when we have sequences, where the sequence elements has a logic order. For intance, a video is a sequence of images indexed in time or a text, a sequence of words with a logic order to give sense to the word sequence. 
So the video or the text can not be input data of this type of neural networks, because each neuron only can store 1 data at the same time, and the sequences depends temporarily from the previous items of the sequence, so we need more than 1 input here.

### Solution
We change the neural network architecture to RNN, now we have another input at the same neuron to modulate the output secuence order, depending indeed on the time `t = 1, t = 2, t = 3,...,t = n`

<p align="center">

<img src="https://github.com/RogerCL24/ChatBot/assets/90930371/ca21162b-46d2-4be2-8893-cb512256b512"/>
</p>

### Explanation

<p align="center">
  
<img src="https://github.com/RogerCL24/ChatBot/assets/90930371/670868b2-e5cf-4153-8874-41cc770aaf55"/>
</p>

Focus on the left neuron, it has an input `X`, a summation `∑`, the activation function and an output `ŷ` up to now it is as always, but now we got another output that feeds back our neuron making it a new input.

Now the right figure which has several neurons, it has not, is the same neuron through the time, as you can observe we have the first input `X(0)` -> _'La'_: so `ŷ(0)` will be _'The'_, ``X(1)`` -> _'manzana'_: `ŷ(1)` will be _'The apple'_ because we have the other input data from before, besides the current one, `X(2)` -> _'roja'_: so `ŷ(2)` will be _'The red apple'_, as you see the order changes due to the grammatical rules, for that reason is important this architecture to follow the sequence, finally `X(3)` -> _'vuela'_: so `ŷ(3)` will be _'The red apple flies'_, cause the feed back input returns _'The red apple'_.

That was only 1 neuron, in a layer would be like this.

<p align="center">
  
<img src="https://github.com/RogerCL24/ChatBot/assets/90930371/076f2dc2-5c8d-42fe-bb0e-450e728debe8"/>
</p>

Same functionality but with more neurons to improve performance, obviously the data used as feed back is stored in memory in order to keep it for further inputs, that produces a problem of memory with long sequences, there are too many words to store and we can not save them all, **solution** -> Transformers ⬇️

## Transformers

As a means to solve the former problem of memory loss, Google published the paper _Attention all you need_ at 2017, this paper basically put forward the new model architecture - **The transformer** - .

<p align="center">
  
<img src="https://github.com/RogerCL24/ChatBot/assets/90930371/3b441e25-3b3e-4ad1-896d-340703af74f4"/>

</p>

 The tranformer is a deep learning (DL) model, based on a ``self-attention`` mechanism that weights the importance of each part of the input data differently. It is mainly used in computer vision (CV) and natural language processing (NLP).

 NLP's Transformer is a new architecture that aims to solve tasks sequence-to-sequence while easily handling long-distance dependencies (memory loss). Computing the input and output representations without using sequence-aligned RNNs or convolutions and it relies entirely on ``self-attention``.

### Blocks
The Transformer is divided in 3 main blocks, and 2 of them has the most important part of the architecture, the ``self-attention`` blocks.

<p align="center">
  
<img src="https://github.com/RogerCL24/ChatBot/assets/90930371/cd6d8a4b-d403-4b90-b4c1-0fc54cfb8ac4"/>

</p>




- 🟦 **Input embeddings**:

<p align="center">
  
<img src="https://github.com/RogerCL24/ChatBot/assets/90930371/5325d47d-465f-49bd-8318-e5e06f6f0d10"/>

</p>

  - As you can see this is a Supervised ML type, we have **X** input data and **Y** output data or labeled values, the inputs goes to the encoder and the outputs goes to the decoder, both of them go through an embedding block and a positional block -> For example, let's do a translation of _'I love fast food'_:

<p align="center">
  
<img src="https://github.com/RogerCL24/ChatBot/assets/90930371/7cbf7a05-981b-4f14-a215-202f0bfd82d3"/>

</p>

  - This words will be the input data to the embedding block, like it was explained previously the embedding block converts the words to vectors (tokens).

<p align="center">
  
<img src="https://github.com/RogerCL24/ChatBot/assets/90930371/04084198-b8cd-46bc-a80c-8e3dd638dbf2"/>
</p>

  - Then that tokens will go to the _positional encoder_, which gives to each word his position in the whole sequence, if we had a longer sentence like 'I like fast food because...' this block is in charge to position 'I' at the first place of the sequence, to do that it applies mathematical functions, cos & sen to the tokens, basically it gives to the system the infomation about in what position each word has been placed.

<p align="center">
  
<img src="https://github.com/RogerCL24/ChatBot/assets/90930371/e5ae7070-5b63-4e74-ad40-cc3747fadaf8"/>
</p>

  - And finally that _green_ sequence goes as input to the encoder and decoder.

<p align="center">
  
<img src="https://github.com/RogerCL24/ChatBot/assets/90930371/b21a82b3-824a-4c8c-a28c-9c5a5e105f43"/>
</p>


- 🟨 **Encoder**:

<p align="center">
  
<img src="https://github.com/RogerCL24/ChatBot/assets/90930371/1d15a04e-f343-45c8-92e1-855c9193c792"/>
</p>

### Multi-Head Attention block
Probably the most important block of the Transformer architecture, this block will take as input data the tokens (already positioned) from the embedding output.

<p align="center" >
  
  <img src="https://github.com/RogerCL24/ChatBot/assets/90930371/b17f03b5-d96d-4d58-9666-4c28a6469957"/>
</p>

- This tokens will go through 3 different Neural Networks, each one is trained to give 3 different vectors -> `Queries`, `Keys` & `Values`.

- The `Queries` vectors are related with the `Keys` vectors.

<p align="center"> 

<img src="https://github.com/RogerCL24/ChatBot/assets/90930371/ef805134-c953-437d-818d-597584b388df"/>
</p>

- I meant, that we have to found which ``queries`` has more likeliness with each `key`, namely, what words have a context more similar than the other words, in order to match correctly the sequence.

- This procedure is done by the _attention matrix_, this matrix is a cross product between the two vectors (query and key), as we know the result of a cross product is a numerical measure of how similar are those vectors spatially.

- After that cross product we apply a Softmax function to determine the probabilities that one word is related to another.

<p align="center"> 

<img src="https://github.com/RogerCL24/ChatBot/assets/90930371/a7d8ccdd-e620-4682-8698-d51781c7456b"/>
</p>

- As they are probabilities the max value of each tuple is 1, that is, for example `I` has _0.9_ at **I** column, makes sense, _0.05_ at **Love** column, is the second bigger value because the most rational combiantion is `I love`, rather than `I fast` or `I food`, then we have _0.03_ at **Fast** column and finally _0.02_ at **Food** column in total -> 0.9 + 0.05 + 0.03 + 0.02 = 1 <sub> You can check each tuple and it will give 1 as well </sub>, this can be known as scoring matrix.

<p align="center">

<img src="https://github.com/RogerCL24/ChatBot/assets/90930371/d2ce512b-2358-4753-8563-d11ae275c775"/>
</p>

- Finally we multiply that scoring matrix with the token `Values`, this will generate the tokens which represent the attention of each word regarding the other words of the sequence.

### Add & Normalize + Feed Forward + Add & Normalize blocks

<p align="center"> 

<img src="https://github.com/RogerCL24/ChatBot/assets/90930371/80c4c570-09fe-4a5a-9526-3c3553adcc97"/>
</p>

- Now we have 2 tokens as input to the Add & Norm, the Attention block output and the positional encoding output as a residual connection, this residual connections (there are 5) are actually done to ensure there is a stronger information signal that flows through deep networks, in fact this is required because in backpropagation you can notice that there are vanishing gradients, which means that at sometime there will be a case where as you keep backpropagating the gradient update becomes 0 (there is not correction of the weight value) and the model stops learning, in order to prevent that we induce more stronger signals from the input in different parts of the network.

<p align="center">

<img src="https://github.com/RogerCL24/ChatBot/assets/90930371/f789ba04-60b5-4488-8915-fb5742c0ab0f"/>
</p>

- This 2 tokens will be added and normalized, the normalization is a layer which each neuron connected to the network of the layer instead of use activations of a large range of positve and negative values normalization encapsulates this values within a smaller range, tipically centered around 0, this allow us to have a much more stable training as during the backpropagation phase and we actually perform a gradient step we are taking much more even steps so it is now easier to learn and hence it is faster, and now we can get to the optimal position or the optimal parameter values more consistently.

<p align="center"> 

<img src="https://github.com/RogerCL24/ChatBot/assets/90930371/2ea9c7e4-e5a6-47a8-a5fd-e223aa85dab4"/>
</p>

Finally, once we have our token(matrix) normalized is easier to learn so we pass it to a Feed Forward Neural Network to learn/train, then we normalize again the token in order to remove workload to the last main block the `Decoder`.

- 🟩 **Decoder**:
<p align="center"> 
  
<img src="https://github.com/RogerCL24/ChatBot/assets/90930371/d102604d-7475-4344-b4fc-61e00fb11153"/>
</p>

- The decoder block is similar to the encoder block, except it calculates the source-target attention.

<p align="center">

<img src="https://github.com/RogerCL24/ChatBot/assets/90930371/7ab81efd-8fd3-4359-a198-3d7ecfbfc817"/>
</p>

- The input of the decoder is the ouput data (the sentence already translated) in order to train, the rest is the same as the encoder except the ``softmax`` function, now softmax has a masking operation for the scoring matrix.

<p align="center"> 

<img src="https://github.com/RogerCL24/ChatBot/assets/90930371/c837ee88-6130-4415-ae90-66e0f341737e"/>
</p>

- As we are applying a training to our network model with unknown words (the words already translated), therefore if we are in the `La` word the network can only use the words that has already seen, namely, `Amo` & `la`.

<p align="center">

<img src="https://github.com/RogerCL24/ChatBot/assets/90930371/0690356f-2e79-4471-8b23-2fb47d845f03"/>
</p>

- For that reason the name of the attention blocs has the word **Masked**, we are masking the unkown words with a 0 and unmaskig the known words with another value, that gives us a lower diagonal matrix.

<p align="center">

<img src="https://github.com/RogerCL24/ChatBot/assets/90930371/b051b882-d701-4a75-8b1a-136f090a16a8"/>
</p>

- The next block (Multi-head attention) executes the same steps as the encoder attention block, but with different inputs (3), 1 will be the output of the encoder (features) that goes to the values vectors and the other 2 are the output from the masked attention block because they are the words we are going to train with.

### Final layer

<p align="center">

<img src="https://github.com/RogerCL24/ChatBot/assets/90930371/02f232ea-eb28-43bb-b5ea-389ebbef10ad"/>
</p>

- Finally, and passing by the feed forward neural network, we have our model trained, the output of the decoder will go through a neural network with a softmax layer, the softmax funciton will find all the probabilities of the ouput vector, that means, if our output vector is the whole dictionary, the position (word) where we have the largest numeric value means that it has the highest probability to be the next word in the sequence or the word to be translated.

<p align="center">

<img src="https://github.com/RogerCL24/ChatBot/assets/90930371/58457629-ccb2-4464-9a57-076cd32442a0"/>
</p>



