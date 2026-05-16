# Quick Reference - Adala Labeling with Evaluation

## 🚀 Quick Start (Choose One)

### Option 1: Run Python Script
```bash
python sentiment_labeling_example.py
```

### Option 2: Run Jupyter Notebook
```bash
jupyter notebook sentiment_labeling_notebook.ipynb
```

### Option 3: Try Multiple Use Cases
```bash
python advanced_labeling_examples.py
```

---

## 📁 File Guide

| File | Purpose | Type |
|------|---------|------|
| `sentiment_labeling_example.py` | Complete sentiment classification pipeline | Script |
| `sentiment_labeling_notebook.ipynb` | Interactive step-by-step tutorial | Notebook |
| `adala/evaluation.py` | Reusable evaluation module | Module |
| `advanced_labeling_examples.py` | 3 additional use cases | Script |
| `IMPLEMENTATION_SUMMARY.md` | Detailed overview | Doc |
| `LABELING_USE_CASE_GUIDE.md` | Complete guide | Doc |

---

## 🎯 Basic Workflow

```python
# 1. Create agent
agent = Agent(
    skills=ClassificationSkill(
        name='sentiment_classification',
        input_template='Review: {review}',
        output_template='Sentiment: {predicted_sentiment}',
        labels={'predicted_sentiment': ["Positive", "Negative", "Neutral"]},
    ),
    environment=StaticEnvironment(
        df=training_data,
        ground_truth_columns={'predicted_sentiment': 'sentiment'}
    )
)

# 2. Train
agent.learn()

# 3. Predict
predictions = agent.run(test_data)

# 4. Evaluate
evaluator = ClassificationEvaluator(ground_truth, predictions)
reporter = MetricsReporter(evaluator)
reporter.print_full_report()
```

---

## 📊 Key Metrics at a Glance

| Metric | Formula | What It Means |
|--------|---------|---------------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| **Precision** | TP/(TP+FP) | Of predictions, how many were correct? |
| **Recall** | TP/(TP+FN) | Of actual positives, how many were found? |
| **F1-Score** | 2(P·R)/(P+R) | Balance between precision & recall |
| **Macro F1** | Average F1 across classes | Equal weight per class |
| **Weighted F1** | F1 weighted by support | Accounts for class imbalance |

---

## ✅ Sentiment Data Format

```python
training_data = pd.DataFrame([
    {"review": "Amazing product!", "sentiment": "Positive"},
    {"review": "Terrible quality.", "sentiment": "Negative"},
    {"review": "It's okay.", "sentiment": "Neutral"},
])
```

---

## 🔍 Understanding Output

### Confusion Matrix
```
          Predicted
          Pos  Neg  Neu
Actual Pos  4   0    1
       Neg  0   3    0
       Neu  1   0    2
```
- Diagonal = Correct predictions
- Off-diagonal = Errors

### Metrics Example
```
Accuracy: 0.8000 (80% of predictions correct)
Precision: 0.7500 (75% of positive predictions were actually positive)
Recall: 0.7500 (75% of actual positives were predicted as positive)
F1: 0.7500 (Balanced measure)
```

---

## 🛠️ Common Customizations

### Use Your Own Data
```python
def create_training_data():
    return pd.DataFrame([
        {"text": "YOUR_TEXT", "label": "YOUR_LABEL"},
        # ...
    ])
```

### Change Labels
```python
labels={'predicted_sentiment': [
    "Excellent",  # Change these
    "Good",
    "Poor"
]}
```

### Change LLM
```python
agent = Agent(
    skills=...,
    default_runtime='openai',  # or other runtime
)
```

---

## 📈 Metrics to Export

```python
# JSON Export
reporter.export_to_json('metrics.json')

# CSV Export  
reporter.export_to_csv('results.csv')

# Get metrics dict
results = evaluator.evaluate()
results.accuracy
results.f1_macro
results.per_class_metrics
results.confusion_matrix
```

---

## 🐛 Common Issues & Fixes

| Error | Fix |
|-------|-----|
| `OpenAI API key not found` | `export OPENAI_API_KEY="sk-..."` |
| `ImportError: No module named 'adala'` | `pip install adala` |
| Predictions are `"Unknown"` | Check output template label matches |
| Low accuracy | Add more training samples |
| `ModuleNotFoundError: sklearn` | `pip install scikit-learn` |

---

## 📊 Sample Output Structure

```
OVERALL METRICS
├─ Accuracy: 0.8000
├─ Macro F1: 0.7500
├─ Weighted F1: 0.8000
└─ Cohen's Kappa: 0.7071

PER-CLASS METRICS
├─ Positive: P=1.0, R=0.67, F1=0.80
├─ Negative: P=0.50, R=1.0, F1=0.67
└─ Neutral: P=0.75, R=0.75, F1=0.75

CONFUSION MATRIX
├─ True Positives (diagonal): 9
├─ False Positives (errors): 1
└─ False Negatives (errors): 1

ERROR ANALYSIS
├─ Total Errors: 2
├─ Error Rate: 0.20 (20%)
└─ Error Types:
    - Positive → Neutral: 1
    - Negative → Positive: 1
```

---

## 💡 When to Use Which Metric

| Situation | Best Metric |
|-----------|-------------|
| Equal class importance | **Macro F1** |
| Imbalanced classes | **Weighted F1** |
| Minimize false alarms | **Precision** |
| Minimize false negatives | **Recall** |
| Overall correctness | **Accuracy** |
| Multi-class balanced | **Cohen's Kappa** |

---

## 🎓 Workflow Checklist

- [ ] Install: `pip install adala scikit-learn pandas matplotlib seaborn`
- [ ] Set API key: `export OPENAI_API_KEY="your-key"`
- [ ] Create training data with ground truth labels
- [ ] Create test data separate from training
- [ ] Initialize Adala agent with ClassificationSkill
- [ ] Train agent with `agent.learn()`
- [ ] Make predictions with `agent.run(test_data)`
- [ ] Evaluate with `ClassificationEvaluator`
- [ ] Print report with `MetricsReporter.print_full_report()`
- [ ] Export metrics with `export_to_json()` or `export_to_csv()`
- [ ] Analyze confusion matrix for error patterns
- [ ] Iterate: Add more training data, adjust labels, etc.

---

## 📚 File Locations

```
/Users/jungheekim/Desktop/Adala/
├── sentiment_labeling_example.py        ← START HERE
├── sentiment_labeling_notebook.ipynb    ← OR HERE
├── advanced_labeling_examples.py        ← OR TRY THIS
├── adala/
│   └── evaluation.py                    ← Reusable module
├── IMPLEMENTATION_SUMMARY.md            ← Full details
├── LABELING_USE_CASE_GUIDE.md          ← Complete guide
└── QUICK_REFERENCE.md                   ← THIS FILE
```

---

## 🔗 Key Classes

### ClassificationEvaluator
```python
evaluator = ClassificationEvaluator(
    ground_truth=['A', 'B', 'A'],
    predictions=['A', 'B', 'B'],
    labels=['A', 'B']
)
results = evaluator.evaluate()
```

### MetricsReporter
```python
reporter = MetricsReporter(evaluator)
reporter.print_full_report()
reporter.export_to_json('metrics.json')
```

### Agent (Adala)
```python
agent = Agent(
    skills=ClassificationSkill(...),
    environment=StaticEnvironment(...),
)
agent.learn()
predictions = agent.run(test_data)
```

---

## 🎯 Tips for Success

1. **Quality data matters**: More diverse training examples → better results
2. **Clear labels**: Use consistent, descriptive label names
3. **Balanced training**: Try to have similar examples per class
4. **Separate test data**: Don't test on training data
5. **Check confusion matrix**: See which classes are commonly confused
6. **Use appropriate metrics**: Different metrics for different goals
7. **Iterate**: Improve by adding more training data

---

## ⚡ One-Liner Examples

```python
# Quick evaluation
evaluator = ClassificationEvaluator(truth, preds); print(f"F1: {evaluator.evaluate().f1_macro}")

# Full report
MetricsReporter(evaluator).print_full_report()

# Export all
reporter.export_to_json('metrics.json'); reporter.export_to_csv('results.csv')

# Get per-class metrics
results = evaluator.evaluate(); print(results.per_class_metrics)
```

---

**Last Updated**: May 16, 2026  
**Version**: 1.0  
**Status**: Ready to Use ✅
