# Adala Sentiment Analysis Labeling - Complete Use Case Guide

This guide demonstrates how to use **Adala** (Autonomous Data Labeling Agent) for sentiment analysis labeling with comprehensive evaluation metrics.

## 📋 Overview

**Use Case**: Customer Review Sentiment Classification
- **Task**: Classify customer reviews into 3 sentiment categories: Positive, Negative, Neutral
- **Goal**: Demonstrate autonomous labeling with ground truth training data and comprehensive evaluation

## 📁 Resources Created

### 1. **sentiment_labeling_example.py** - Complete Standalone Script
A full-featured Python script that implements the entire workflow:

```python
# Features:
- Training data creation with 15 labeled reviews
- Test data creation with 10 reviews
- Agent initialization and training
- Predictions on test data
- Comprehensive evaluation metrics
- Error analysis and visualizations
```

**Run it**:
```bash
cd /Users/jungheekim/Desktop/Adala
python sentiment_labeling_example.py
```

**What it does**:
1. Creates training/test datasets
2. Initializes Adala agent with ClassificationSkill
3. Trains agent on ground truth data
4. Makes predictions on test set
5. Calculates comprehensive metrics (accuracy, precision, recall, F1)
6. Generates confusion matrix
7. Analyzes errors
8. Creates visualizations
9. Exports results to JSON

---

### 2. **sentiment_labeling_notebook.ipynb** - Interactive Jupyter Notebook
Interactive notebook with step-by-step sections:

**Sections**:
1. **Import Required Libraries** - Import Adala and data science libraries
2. **Load or Create Dataset** - Create training and test data
3. **Initialize Adala Pipeline** - Setup sentiment classification agent
4. **Generate Labels** - Train agent and make predictions
5. **Evaluate Outputs** - Calculate performance metrics
6. **Visualize Results** - Create comprehensive charts and tables

**Run it**:
```bash
jupyter notebook /Users/jungheekim/Desktop/Adala/sentiment_labeling_notebook.ipynb
```

---

### 3. **adala/evaluation.py** - Reusable Evaluation Module
Modular evaluation utilities for any labeling task:

**Classes**:
- `ClassificationEvaluator`: Calculate metrics, confusion matrix, error analysis
- `MetricsReporter`: Generate formatted reports and export results

**Key Methods**:
- `evaluate()` - Calculate all metrics
- `get_confusion_matrix_df()` - Get confusion matrix as DataFrame
- `get_misclassified_samples()` - Get list of errors
- `print_full_report()` - Print comprehensive report
- `export_to_json()` - Export metrics to JSON
- `export_to_csv()` - Export detailed results to CSV

**Usage Example**:
```python
from adala.evaluation import ClassificationEvaluator, MetricsReporter

# After getting predictions
ground_truth = ['Positive', 'Negative', 'Neutral', ...]
predictions = ['Positive', 'Positive', 'Neutral', ...]

# Evaluate
evaluator = ClassificationEvaluator(ground_truth, predictions)
reporter = MetricsReporter(evaluator)

# Print and export
reporter.print_full_report()
reporter.export_to_json('/path/to/metrics.json')
reporter.export_to_csv('/path/to/results.csv')
```

---

## 📊 Evaluation Metrics Explained

### Overall Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Percentage of correct predictions |
| **Cohen's Kappa** | Agreement accounting for chance (better for imbalanced data) |
| **Matthews Corr** | Correlation coefficient (-1 to 1) |

### Averaged Metrics

| Metric | Description |
|--------|-------------|
| **Macro-averaged** | Unweighted average across all classes (same weight per class) |
| **Weighted** | Average weighted by class support (accounts for class imbalance) |

### Per-Class Metrics

For each class (Positive/Negative/Neutral):
- **Precision**: Of predicted positives, how many were actually positive?
- **Recall**: Of actual positives, how many were predicted as positive?
- **F1-Score**: Harmonic mean of precision and recall

### Confusion Matrix
Shows:
- **True Positives (TP)**: Correctly predicted positive
- **False Positives (FP)**: Incorrectly predicted as positive
- **False Negatives (FN)**: Missed positive (predicted as other)
- **True Negatives (TN)**: Correctly predicted as other

---

## 🚀 Quick Start

### Step 1: Run the Standalone Script
```bash
cd /Users/jungheekim/Desktop/Adala
python sentiment_labeling_example.py
```

This will:
- Train the agent
- Generate predictions
- Print detailed metrics
- Show error analysis
- Create visualizations

### Step 2: Open the Jupyter Notebook
```bash
jupyter notebook sentiment_labeling_notebook.ipynb
```

Run cells sequentially to see each step of the process.

### Step 3: Adapt to Your Data
Modify the data creation functions to use your own reviews:

```python
# In sentiment_labeling_example.py, modify create_training_data():
def create_training_data() -> pd.DataFrame:
    training_data = pd.DataFrame([
        {"review": "YOUR_REVIEW_1", "sentiment": "Positive"},
        {"review": "YOUR_REVIEW_2", "sentiment": "Negative"},
        # ... add your data
    ])
    return training_data
```

---

## 📈 Sample Output

```
======================================================================
EVALUATION METRICS - SENTIMENT LABELING TASK
======================================================================

--- OVERALL METRICS ---
Accuracy: 0.8000

Macro-averaged (unweighted):
  Precision: 0.7500
  Recall:    0.7500
  F1-score:  0.7500

Weighted (by support):
  Precision: 0.8000
  Recall:    0.8000
  F1-score:  0.8000

--- PER-CLASS METRICS ---

Positive:
  Precision: 1.0000
  Recall:    0.6667
  F1-score:  0.8000

Negative:
  Precision: 0.5000
  Recall:    1.0000
  F1-score:  0.6667

Neutral:
  Precision: 0.7500
  Recall:    0.7500
  F1-score:  0.7500

--- CONFUSION MATRIX ---
          Positive  Negative  Neutral
Positive         2         0        1
Negative         0         2        0
Neutral          0         1        2
```

---

## 🔧 Customization Guide

### Change the Classification Task
Modify the `ClassificationSkill` in the agent initialization:

```python
agent = Agent(
    skills=ClassificationSkill(
        name='your_task_name',
        input_template='Your input: {input_field}',
        output_template='Your output: {your_label_field}',
        labels={
            'your_label_field': [
                "Label1",
                "Label2",
                "Label3"
            ]
        },
    ),
    # ... rest of config
)
```

### Add More Training Samples
Add more rows to `create_training_data()`:

```python
training_data = pd.DataFrame([
    {"review": "...", "sentiment": "..."},
    # ... more samples
])
```

### Change the LLM Runtime
By default uses OpenAI. To use a different runtime:

```python
from adala.runtimes._openai import OpenAIChatRuntime

agent = Agent(
    skills=...,
    environment=...,
    default_runtime='your-runtime'  # Change this
)
```

---

## 📝 Troubleshooting

### Issue: "OpenAI API key not found"
**Solution**: Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

### Issue: Agent predictions all say "Unknown"
**Solution**: Check that output template label name matches skill labels:
```python
output_template='Sentiment: {predicted_sentiment}'
labels={'predicted_sentiment': [...]}  # Must match!
```

### Issue: Low accuracy
**Solution**:
1. Add more training samples
2. Make training examples more diverse
3. Use clearer, more descriptive labels
4. Try different LLM models

---

## 📚 Adala Documentation

- **Official Repo**: https://github.com/HumanSignal/Adala
- **Skills Available**:
  - ClassificationSkill
  - ExtractionSkill
  - QASkill
  - SummarizationSkill
  - TranslationSkill
  - RAGSkill

- **Environments**:
  - StaticEnvironment (ground truth data)
  - CodeEnvironment (code execution)
  - WebEnvironment
  - KafkaEnvironment

---

## 🎯 Use Cases to Explore

1. **Text Classification**: Sentiment, topic, spam detection
2. **Entity Extraction**: Extract names, locations, dates
3. **Question Answering**: Answer questions from context
4. **Summarization**: Generate summaries of long texts
5. **Translation**: Translate between languages
6. **Document Labeling**: Categorize documents

---

## 💡 Best Practices

1. **Quality Training Data**: Use diverse, representative examples
2. **Clear Labels**: Use consistent, descriptive label names
3. **Balanced Data**: Aim for roughly equal examples per class
4. **Test Data**: Keep separate test data for evaluation
5. **Metrics**: Use appropriate metrics for your task (F1 for imbalanced data)
6. **Error Analysis**: Always review misclassified examples
7. **Iteration**: Improve by adding more training data

---

## 📞 Support

For issues with:
- **Adala**: Check GitHub issues and documentation
- **Metrics**: Review sklearn documentation
- **This code**: Review the scripts and notebook comments

---

Created: May 2026
Last Updated: May 16, 2026
