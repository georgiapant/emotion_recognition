# Emotion Recognition

### This repo currently includes:

- Five different approaches to emotion recognition. most of which include stacking on BERT output other layers
  - Bi-LSTM layer
  - Combining the last four hidden layers and passing them through Bi-GRU layer
  - Ensembing two pre-trained models (BERT and RoBERTa)
  - Creating extra emotional features from lexicon resources (VAD and NRC) and using Bi-LSTM and attention net
  - Creating a selective predictor that does not make a prediction for instances without enough confidence

For all models the GoEmotions dataset was used modifying the labels to the EKMAN emotions same as in the GoEmotions introduction paper
