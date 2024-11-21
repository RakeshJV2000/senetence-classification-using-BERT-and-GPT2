# Senetence Classification Using BERT and GPT2
This project implements and evaluates various text classification approaches using BERT and GPT-2 transformers on the DBPedia dataset. The implementation utilizes PyTorch and HuggingFace Transformers, featuring a ReLU-activated 2-layer perceptron classifier (768→32→14) for document classification. The architecture employs different feature extraction techniques including BERT's [CLS] token, mean pooling, and max pooling of token representations. Key hyperparameters include a mini-batch size of 32, a learning rate of 5e-4 with Adam optimizer, and training for one epoch. The experiments are conducted with frozen transformer features and evaluated across 5 different random seeds to ensure reliability. The project also explores the fine-tuning capabilities of BERT's final layers and investigates GPT-2's effectiveness as a feature extractor, providing comprehensive comparisons of different approaches for text classification tasks.

## Classification with CLS token embeddings from BERT
- Mean Development Accuracy: 96.16% ± 0.95%
- Best Test Accuracy (from best dev model): 97.70%

## Classification with Mean Pooling BERT embeddings
- Mean Development Accuracy: 96.96% ± 0.22%
- Best Test Accuracy (from best dev model): 97.10%

## Classification with Max Pooling BERT embeddings
- Mean Development Accuracy: 65.18% ± 4.68%
- Best Test Accuracy (from best dev model): 70.40%

## Analysis
The experimental results demonstrate significant variations in performance across different pooling mechanisms when using frozen BERT features. Mean pooling emerged as the most effective approach, achieving the highest development accuracy of 96.96% with remarkably low variance (±0.22%), indicating stable and reliable performance across different random initializations. The CLS token method performed comparably well, with a development accuracy of 96.16% (±0.95%), though showing slightly higher variance. Notably, while the CLS token method achieved the highest test accuracy (97.70%) on its best model, its higher standard deviation suggests less consistent performance. In contrast, max pooling significantly underperformed, with a development accuracy of only 65.18% and high variance (±4.68%), suggesting this method may not effectively capture the semantic information needed for classification. The substantial performance gap between max pooling and the other two methods indicates that averaging or using dedicated classification tokens better preserves the relevant features for text classification tasks. Based on these results, mean pooling appears to be the most robust choice for this specific classification task, offering the best balance between performance and stability.

## Finetuning BERT's layer 10 and layer 11

- Mean Development Accuracy: 99.12% ± 0.26%
- Best Test Accuracy (from best dev model): 99.70%

Fine-tuning BERT's last two layers significantly outperformed all other approaches, achieving the highest mean development accuracy of 99.12% with remarkable stability (±0.26%). This represents a substantial improvement over the frozen feature methods: CLS token (96.16% ± 0.95%), mean pooling (96.96% ± 0.22%), and max pooling (65.18% ± 4.68%). The fine-tuned model's best test accuracy of 99.70% also surpassed the test accuracies of CLS token (97.70%), mean pooling (97.10%), and max pooling (70.40%). This superior performance demonstrates that allowing the model to adapt its higher-level features through fine-tuning while keeping lower layers frozen, provides a better balance between feature extraction and task-specific optimization. The low standard deviation also suggests that fine-tuning produces more consistent results across different random initializations compared to the other methods.

## GPT2 as a feature extractor

- Mean Development Accuracy: 23.16% ± 3.88%
- Best Test Accuracy (from best dev model): 28.50%

GPT-2's feature extraction performance was significantly inferior to all other approaches, with a mean development accuracy of only 23.16% (±3.88%) and best test accuracy of 28.50%. This is dramatically lower than the BERT-based methods achieved. The poor performance can be attributed to GPT-2's architectural design focusing on autoregressive language modeling rather than bidirectional understanding, making it less suitable for classification tasks. 
