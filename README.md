
## Graph-based Dependency Parser

The data is in Conll06 format.

The parser components are:
+ Structured Perceptron
+ CLE Decoder
+ Feature Model

### Training
To train, ensure that all arguments in the file ```run_training.sh``` are set as desidered and run script:

```bash run_training.sh```

### Predicting + Evaluating
To make predictions using a trained model, ensure all arguments in the file ```run_experiments.sh``` are set as desired and run script:

```bash run_experiments.sh```

This will create a .pred file in the preds directory.
It will also load the newly created file and evaluate again a gold standard (if not wanted, comment out).

## Results

### English

| N_EPOCHS | LR | INIT | FEATS | UAS | UCM | 
| --- | --- | --- | --- | --- | --- | 
| 3 | 0.3 | zeros | basic | 0.841 | 0.143 | 
| 10 | 0.3 | zeros | basic | 0.857 | 0.173 | 
| 10 | 0.5 | zeros | basic | 0.857 | 0.167 | 
| 10 | 0.5 + decay | zeros | basic | 0.867 | 0.195 | 
| 10 | 0.5 + decay | random | basic | 0.867 | 0.182 |
| 20 + ES 17 | 0.5 + decay | zeros | basic | 0.867 | 0.197 | 
| 20 + ES 14 | 0.5 + decay | zeros | additional | 0.873 | 0.19 | 
| 20 + ES 16 | 0.5 + decay | zeros | all | 0.878 | 0.216 | 

### German

| N_EPOCHS | LR | INIT | FEATS | UAS | UCM | 
| --- | --- | --- | --- | --- | --- | 
| 3 | 0.3 | zeros | basic | 0.867 | 0.336 | 
| 10 | 0.3 | zeros | basic | 0.883 | 0.357 | 
| 10 | 0.5 | zeros | basic | 0.882 | 0.355 | 
| 10 | 0.5 + decay | zeros | basic | 0.894 | 0.385 |
| 10 | 0.5 + decay | random | basic | 0.893 | 0.387 |
| 20 + ES 15 | 0.5 + decay | zeros | basic | 0.894 | 0.381 | 
| 20 + ES 16 | 0.5 + decay | zeros | additional | 0.897 | 0.392 | 
| 20 + ES 13 | 0.5 + decay | zeros | all | 0.904 | 0.416 | 
