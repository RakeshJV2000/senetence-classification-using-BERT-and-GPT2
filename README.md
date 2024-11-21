# senetence-classification-using-BERT-and-GPT2

This project implements and evaluates various text classification approaches using BERT and GPT-2 transformers on the DBPedia dataset. The implementation utilizes PyTorch and HuggingFace Transformers, featuring a ReLU-activated 2-layer perceptron classifier (768→32→14) for document classification. The architecture employs different feature extraction techniques including BERT's [CLS] token, mean pooling, and max pooling of token representations. Key hyperparameters include a mini-batch size of 32, learning rate of 5e-4 with Adam optimizer, and training for one epoch. The experiments are conducted with frozen transformer features and evaluated across 5 different random seeds to ensure reliability. The project also explores fine-tuning capabilities of BERT's final layers and investigates GPT-2's effectiveness as a feature extractor, providing comprehensive comparisons of different approaches for text classification tasks.
