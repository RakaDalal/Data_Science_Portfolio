# English_to_French_translation

I improved the basic encoder-decoder based Sequence to Sequence (Seq2Seq) translator in Keras for English to French translation.

I have made the following improvements to the architecture:

1. I have replaced the LSTM layer in the encoder and decoder with GRU layers respectively. The GRU layer introduced by Cho et. al. 2014 is much more simple and effective and immediately leads to a reduction in validation loss.
2. I have generalized the architecture by constructing deep encoder and deep decoder where the depth of the GRU layers in both of them is taken as an input.
3. I have introduced an additional hidden layer followed by a dropout layer for processing of the output from the decoder GRU. 

All these steps have resulted in a drastic reduction in validation loss (final value is 0.4698 for depth=3) keeping the number of epochs fixed at 5.

