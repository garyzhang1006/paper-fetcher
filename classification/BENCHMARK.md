# Classification benchmark

This classifier applies the workflow from `classification.pdf` and
`classification_tutorial.ipynb` to arXiv papers:

- title and abstract text become TF-IDF features;
- `primary_category` becomes the class label;
- training, validation, and test partitions stay separate;
- the neural network returns raw logits for cross-entropy loss;
- validation macro-F1 selects the checkpoint;
- the untouched test partition measures final quality;
- softmax probabilities appear only in prediction and confidence reports.

## Reproduction

```bash
paper-fetcher-classify \
  --dataset dataset/papers.jsonl \
  --output-dir data/category_classifier \
  --epochs 20 \
  --patience 4 \
  --device cpu
```

Run date: July 23, 2026

Dataset SHA-256:
`1bfe64d02fffedaaa65ef00866e47e8c688966062ba54572c404a08c4657c78a`

Configuration: seed 42, 8,000 maximum features, 384 hidden neurons, 0.25
dropout, batch size 128, and square-root inverse-frequency class weights.

## Observed results

| Metric | Result |
| --- | ---: |
| Included categories | 112 |
| Excluded rare papers | 43 |
| Training papers | 5,394 |
| Validation papers | 1,157 |
| Test papers | 1,157 |
| Best epoch | 18 |
| Test accuracy | 61.37% (710/1,157) |
| Test macro-F1 | 34.65% |
| Test weighted-F1 | 59.15% |
| Test top-3 accuracy | 84.62% |
| Expected calibration error | 4.83% |
| Majority-class test baseline | 10.11% |
| Uniform-random expected accuracy | 0.89% |

Accuracy rises when the classifier abstains on uncertain examples:

| Confidence threshold | Coverage | Accuracy |
| --- | ---: | ---: |
| 0.50 | 61.80% | 76.78% |
| 0.70 | 42.70% | 87.04% |
| 0.90 | 24.98% | 93.08% |

These numbers measure this exact dataset snapshot. They do not guarantee
performance on future papers, unseen categories, or shifted subject areas.
Confidence is a model score, not certainty.
