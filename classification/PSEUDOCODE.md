# Classifier pseudocode

This is the exact logical flow implemented by
`src/arxiv_kg/category_classifier.py`.

## Train, validate, and test

```text
INPUT:
    JSONL papers containing arxiv_id, title, abstract, primary_category

LOAD every paper
REJECT malformed rows and duplicate arxiv_id values
REMOVE categories with fewer than minimum_class_count papers
ENCODE each primary_category as an integer class

FOR each class:
    SHUFFLE its papers with the fixed seed
    PLACE at least one paper in test
    PLACE at least one paper in validation
    PLACE all remaining papers in training

FIT word and character TF-IDF only on training title + abstract
TRANSFORM validation and test with that fitted TF-IDF

CREATE neural network:
    TF-IDF input
    -> linear hidden layer
    -> ReLU
    -> dropout
    -> linear output with one raw logit per category

CALCULATE class weights from training labels only

FOR each epoch:
    SET model to training mode
    FOR each shuffled training batch:
        CLEAR old gradients
        logits = model(features)
        loss = weighted_cross_entropy(logits, correct_categories)
        BACKPROPAGATE loss
        UPDATE weights with AdamW

    SET model to evaluation mode
    DISABLE gradient calculation
    CALCULATE validation predictions and unweighted loss
    CALCULATE validation accuracy and macro-F1

    IF validation macro-F1 improved:
        SAVE model weights as best checkpoint
    ELSE IF patience is exhausted:
        STOP training

RESTORE best validation-selected checkpoint
EVALUATE untouched test papers once
CALCULATE accuracy, macro-F1, weighted-F1, top-3 accuracy, and calibration
RECORD confidence-filtered accuracy, confusion pairs, and mistakes
```

The network returns raw logits during training. Cross-entropy applies the
required internal normalization. Softmax is used after training loss
calculation for evaluation metrics, confidence analysis, and inference. It is
never applied before cross-entropy.

## Save artifacts

```text
SERIALIZE fitted TF-IDF vectorizer
SERIALIZE category labels in exact neural-output order
HASH both serialized artifacts
WRITE model, vectorizer, labels, metrics, curves, and analysis to staging
IF every staged artifact succeeded:
    REPLACE previous artifact directory with complete staged directory
ELSE:
    KEEP previous artifact directory unchanged
```

Hashes stop mixed model, vectorizer, or label files from silently producing
wrong categories. Staging prevents a late save failure from destroying the
previous complete run.

## Predict one new paper

```text
LOAD model, TF-IDF vectorizer, and category labels
IF artifact version is 1:
    VERIFY vectorizer and label hashes match model checkpoint
ELSE IF artifact version is unknown:
    REJECT artifact
ELSE:
    LOAD legacy artifact and verify category count
COMBINE new title and abstract
TRANSFORM text using training-fitted TF-IDF
COMPUTE raw neural-network logits
APPLY softmax to obtain display probabilities
RETURN highest-probability categories in descending order
```

Confidence is a model score. It is not proof that a prediction is correct.
