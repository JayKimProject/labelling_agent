# Databricks notebook source

# MAGIC %md
# # Labelling Leak Use Cases with Adala
#
# This notebook demonstrates common **label leakage** scenarios in data labeling pipelines
# and how to detect, diagnose, and mitigate them using Adala.
#
# ## What is Label Leakage?
# Label leakage occurs when information that would not be available at inference time
# leaks into the training or evaluation process, producing artificially inflated metrics
# that do not reflect real-world performance.
#
# ## Use Cases Covered
# 1. Ground Truth Leakage — label text appears directly in the input
# 2. Temporal Leakage — future information used to label past events
# 3. Annotator Bias Leakage — consistent annotator errors transfer across splits
# 4. Feature Leakage — a feature column encodes the label
# 5. Data Contamination — test samples duplicated in training set
# 6. Pipeline Leakage — preprocessing fitted on the full dataset including test

# COMMAND ----------

# Section 1: Imports

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("LABELLING LEAK USE CASES - ADALA FRAMEWORK")
print("=" * 70)
print("✓ Libraries imported")

# COMMAND ----------

# Section 2: Shared Adala Agent Builder

from adala.agents import Agent
from adala.environments import StaticEnvironment
from adala.skills import ClassificationSkill


def build_sentiment_agent(training_df: pd.DataFrame) -> Agent:
    """Build a sentiment classification agent trained on the given dataframe."""
    return Agent(
        skills=ClassificationSkill(
            name='sentiment_classification',
            input_template='Review: {review}',
            output_template='Sentiment: {predicted_sentiment}',
            labels={
                'predicted_sentiment': ["Positive", "Negative", "Neutral"]
            },
        ),
        environment=StaticEnvironment(
            df=training_df,
            ground_truth_columns={'predicted_sentiment': 'sentiment'}
        )
    )


def evaluate(ground_truth: List[str], predictions: List[str], label: str) -> Dict:
    acc = accuracy_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions, average='weighted', zero_division=0)
    print(f"\n[{label}]")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 (wtd) : {f1:.4f}")
    return {'accuracy': acc, 'f1': f1}


def run_agent(agent: Agent, test_df: pd.DataFrame) -> List[str]:
    agent.learn()
    output = agent.run(test_df)
    return [
        row.get('predicted_sentiment', 'Unknown') if isinstance(row, dict) else str(row)
        for row in output
    ]

# COMMAND ----------

# MAGIC %md
# ## Use Case 1: Ground Truth Leakage
#
# **Scenario**: The label value is embedded directly in the review text, either by a
# data-export bug or a poorly designed template. The model learns to copy the keyword
# rather than understand sentiment.
#
# **Detection**: Near-perfect training accuracy with a massive drop when the keyword
# is stripped from the input.

# COMMAND ----------

print("\n" + "=" * 70)
print("USE CASE 1: GROUND TRUTH LEAKAGE")
print("=" * 70)

# Leaked dataset: the word "POSITIVE/NEGATIVE/NEUTRAL" was accidentally prepended
# to the review text during the data-export step.
leaked_training = pd.DataFrame([
    {"review": "[POSITIVE] Amazing product! Exceeded all expectations.", "sentiment": "Positive"},
    {"review": "[NEGATIVE] Terrible quality. Very disappointed.", "sentiment": "Negative"},
    {"review": "[NEUTRAL] It's okay, nothing special.", "sentiment": "Neutral"},
    {"review": "[POSITIVE] Love it! Highly recommend to everyone.", "sentiment": "Positive"},
    {"review": "[NEGATIVE] Waste of money. Complete disappointment.", "sentiment": "Negative"},
    {"review": "[NEUTRAL] Good product, but could be better.", "sentiment": "Neutral"},
    {"review": "[POSITIVE] Absolutely fantastic! Worth every penny.", "sentiment": "Positive"},
    {"review": "[NEGATIVE] Poor customer service and bad quality.", "sentiment": "Negative"},
    {"review": "[NEUTRAL] Average product, does what it's supposed to.", "sentiment": "Neutral"},
])

# Clean test set: no leaked tags
clean_test = pd.DataFrame([
    {"review": "Fantastic quality and fast delivery!", "sentiment": "Positive"},
    {"review": "Defective product upon arrival.", "sentiment": "Negative"},
    {"review": "Met expectations, satisfactory.", "sentiment": "Neutral"},
    {"review": "Outstanding! My family loves it.", "sentiment": "Positive"},
    {"review": "Not worth the price. Cheap materials.", "sentiment": "Negative"},
])

# Train on leaked data
leaked_agent = build_sentiment_agent(leaked_training)
leaked_preds = run_agent(leaked_agent, clean_test)

print("\n--- LEAKED TRAINING EXAMPLES ---")
print(leaked_training[['review', 'sentiment']].head(4).to_string(index=False))

evaluate(clean_test['sentiment'].tolist(), leaked_preds, "Trained on leaked data vs. clean test")

# Mitigation: strip the tag before training
clean_training = leaked_training.copy()
clean_training['review'] = clean_training['review'].str.replace(
    r'\[(POSITIVE|NEGATIVE|NEUTRAL)\] ', '', regex=True
)
clean_agent = build_sentiment_agent(clean_training)
clean_preds = run_agent(clean_agent, clean_test)

evaluate(clean_test['sentiment'].tolist(), clean_preds, "After mitigation (tag stripped)")

print("\n⚠  Diagnosis: If accuracy drops sharply after removing the tag,")
print("   the model learned from the leaked keyword, not from semantics.")

# COMMAND ----------

# MAGIC %md
# ## Use Case 2: Temporal Leakage
#
# **Scenario**: A customer-churn labeling pipeline uses the final churn outcome (recorded
# months later) as a feature in the same row that is meant to predict churn risk.
# At inference time this outcome column is not yet known.
#
# **Detection**: Extremely high accuracy that collapses to near-random when the
# future-dated column is removed.

# COMMAND ----------

print("\n" + "=" * 70)
print("USE CASE 2: TEMPORAL LEAKAGE")
print("=" * 70)

# The column `final_status` is recorded after the event window closes.
# It should never appear as a feature at prediction time.
churn_training_leaked = pd.DataFrame([
    {"review": "Haven't logged in for 30 days", "final_status": "churned",    "churn_label": "Churn"},
    {"review": "Opened 5 support tickets this week", "final_status": "churned", "churn_label": "Churn"},
    {"review": "Upgraded plan last week",            "final_status": "active",  "churn_label": "No Churn"},
    {"review": "Downloaded reports daily",           "final_status": "active",  "churn_label": "No Churn"},
    {"review": "Requested account deletion",         "final_status": "churned", "churn_label": "Churn"},
    {"review": "Invited 3 teammates this month",     "final_status": "active",  "churn_label": "No Churn"},
])

churn_test_leaked = pd.DataFrame([
    {"review": "Stopped using the dashboard",        "final_status": "churned", "churn_label": "Churn"},
    {"review": "Expanded to premium tier",           "final_status": "active",  "churn_label": "No Churn"},
    {"review": "Filed a billing dispute",            "final_status": "churned", "churn_label": "Churn"},
    {"review": "Set up integrations last week",      "final_status": "active",  "churn_label": "No Churn"},
])

# Build a ClassificationSkill that exposes the leaked column
churn_agent_leaked = Agent(
    skills=ClassificationSkill(
        name='churn_classification',
        # final_status is visible — this leaks the answer
        input_template='Activity: {review}  Account status: {final_status}',
        output_template='Prediction: {predicted_churn}',
        labels={'predicted_churn': ["Churn", "No Churn"]},
    ),
    environment=StaticEnvironment(
        df=churn_training_leaked,
        ground_truth_columns={'predicted_churn': 'churn_label'}
    )
)
churn_agent_leaked.learn()
leaked_churn_output = churn_agent_leaked.run(churn_test_leaked)
leaked_churn_preds = [
    r.get('predicted_churn', 'Unknown') if isinstance(r, dict) else str(r)
    for r in leaked_churn_output
]

# Mitigation: remove the future column from the template
churn_agent_clean = Agent(
    skills=ClassificationSkill(
        name='churn_classification',
        input_template='Activity: {review}',
        output_template='Prediction: {predicted_churn}',
        labels={'predicted_churn': ["Churn", "No Churn"]},
    ),
    environment=StaticEnvironment(
        df=churn_training_leaked[['review', 'churn_label']],
        ground_truth_columns={'predicted_churn': 'churn_label'}
    )
)
churn_agent_clean.learn()
clean_churn_output = churn_agent_clean.run(churn_test_leaked[['review']])
clean_churn_preds = [
    r.get('predicted_churn', 'Unknown') if isinstance(r, dict) else str(r)
    for r in clean_churn_output
]

gt_churn = churn_test_leaked['churn_label'].tolist()
evaluate(gt_churn, leaked_churn_preds, "With temporal leakage (final_status visible)")
evaluate(gt_churn, clean_churn_preds, "After mitigation (future column removed)")

print("\n⚠  Diagnosis: Remove any column whose value is only known after the")
print("   prediction window closes before passing data to the agent.")

# COMMAND ----------

# MAGIC %md
# ## Use Case 3: Annotator Bias Leakage
#
# **Scenario**: All training data is labeled by Annotator A who consistently mislabels
# ambiguous reviews as "Neutral" instead of "Negative". The same annotator later labels
# part of the test set, so errors are consistent and the agent appears accurate.
# When evaluated against a gold standard, performance collapses.
#
# **Detection**: Large gap between inter-annotator agreement metrics and held-out
# gold-standard accuracy.

# COMMAND ----------

print("\n" + "=" * 70)
print("USE CASE 3: ANNOTATOR BIAS LEAKAGE")
print("=" * 70)

# Annotator A systematically labels mildly negative reviews as Neutral.
# Gold standard (second annotator) correctly labels them as Negative.
biased_training = pd.DataFrame([
    {"review": "Product arrived late and packaging was damaged.", "annotator_a": "Neutral",  "gold": "Negative"},
    {"review": "Customer support took three days to respond.",    "annotator_a": "Neutral",  "gold": "Negative"},
    {"review": "Amazing quality! Absolutely love it.",            "annotator_a": "Positive", "gold": "Positive"},
    {"review": "Terrible. Broke on day one.",                     "annotator_a": "Negative", "gold": "Negative"},
    {"review": "Okay product, a bit overpriced.",                 "annotator_a": "Neutral",  "gold": "Neutral"},
    {"review": "Instructions were confusing and incomplete.",     "annotator_a": "Neutral",  "gold": "Negative"},
    {"review": "Exceeded all my expectations!",                   "annotator_a": "Positive", "gold": "Positive"},
    {"review": "Delivery was slower than promised.",              "annotator_a": "Neutral",  "gold": "Negative"},
])

biased_test = pd.DataFrame([
    {"review": "Missing parts inside the box.",                   "annotator_a": "Neutral",  "gold": "Negative"},
    {"review": "Fantastic, will buy again!",                      "annotator_a": "Positive", "gold": "Positive"},
    {"review": "Not the color I ordered.",                        "annotator_a": "Neutral",  "gold": "Negative"},
    {"review": "Does the job. Nothing special.",                  "annotator_a": "Neutral",  "gold": "Neutral"},
    {"review": "Worst purchase of the year.",                     "annotator_a": "Negative", "gold": "Negative"},
])

# Train on biased labels
biased_agent = Agent(
    skills=ClassificationSkill(
        name='sentiment_classification',
        input_template='Review: {review}',
        output_template='Sentiment: {predicted_sentiment}',
        labels={'predicted_sentiment': ["Positive", "Negative", "Neutral"]},
    ),
    environment=StaticEnvironment(
        df=biased_training.rename(columns={'annotator_a': 'sentiment'}),
        ground_truth_columns={'predicted_sentiment': 'sentiment'}
    )
)
biased_agent.learn()
biased_output = biased_agent.run(biased_test[['review']])
biased_preds = [
    r.get('predicted_sentiment', 'Unknown') if isinstance(r, dict) else str(r)
    for r in biased_output
]

print("\n--- ANNOTATOR BIAS EXAMPLE ---")
print(biased_training[['review', 'annotator_a', 'gold']].to_string(index=False))

# Evaluate against biased labels (inflated) and gold labels (real)
evaluate(biased_test['annotator_a'].tolist(), biased_preds, "vs. Annotator A labels (biased)")
evaluate(biased_test['gold'].tolist(),        biased_preds, "vs. Gold standard labels")

print("\n⚠  Diagnosis: Inter-annotator agreement (Cohen's kappa) < 0.7 suggests")
print("   annotator-specific patterns may be leaking into the model.")
print("   Mitigation: use gold-standard adjudication or majority vote across annotators.")

# COMMAND ----------

# MAGIC %md
# ## Use Case 4: Feature Leakage
#
# **Scenario**: A support-ticket priority labeling task includes a `resolution_time_hours`
# column. Urgent tickets are resolved quickly (priority team responds first); this
# correlation makes the column a near-perfect proxy for the label — but it is recorded
# after the ticket is resolved, so it is unavailable at labeling time.
#
# **Detection**: Feature importance analysis shows one column dominates; removing it
# causes large accuracy drop.

# COMMAND ----------

print("\n" + "=" * 70)
print("USE CASE 4: FEATURE LEAKAGE")
print("=" * 70)

ticket_training_leaked = pd.DataFrame([
    {"ticket": "Server is completely down, revenue impact",   "resolution_time_hours": 0.5,  "priority": "Urgent"},
    {"ticket": "Cannot log in, locked out of account",        "resolution_time_hours": 0.8,  "priority": "Urgent"},
    {"ticket": "How do I export my data?",                    "resolution_time_hours": 48.0, "priority": "Low"},
    {"ticket": "Feature request: dark mode",                  "resolution_time_hours": 72.0, "priority": "Low"},
    {"ticket": "Billing charge appears twice this month",     "resolution_time_hours": 4.0,  "priority": "High"},
    {"ticket": "API rate limit exceeded in production",       "resolution_time_hours": 1.2,  "priority": "Urgent"},
    {"ticket": "Would love a mobile app",                     "resolution_time_hours": 96.0, "priority": "Low"},
    {"ticket": "Dashboard loads slowly for large datasets",   "resolution_time_hours": 12.0, "priority": "Medium"},
])

ticket_test = pd.DataFrame([
    {"ticket": "Complete outage, all users affected",         "resolution_time_hours": 0.3,  "priority": "Urgent"},
    {"ticket": "Password reset email not arriving",           "resolution_time_hours": 2.0,  "priority": "High"},
    {"ticket": "Can you add CSV export?",                     "resolution_time_hours": 60.0, "priority": "Low"},
    {"ticket": "Memory leak in the agent SDK",                "resolution_time_hours": 3.0,  "priority": "High"},
])

# Leaked: resolution_time_hours is in the input template
ticket_agent_leaked = Agent(
    skills=ClassificationSkill(
        name='priority_classification',
        input_template='Ticket: {ticket}  Resolution time (hrs): {resolution_time_hours}',
        output_template='Priority: {predicted_priority}',
        labels={'predicted_priority': ["Urgent", "High", "Medium", "Low"]},
    ),
    environment=StaticEnvironment(
        df=ticket_training_leaked,
        ground_truth_columns={'predicted_priority': 'priority'}
    )
)
ticket_agent_leaked.learn()
leaked_ticket_output = ticket_agent_leaked.run(ticket_test)
leaked_ticket_preds = [
    r.get('predicted_priority', 'Unknown') if isinstance(r, dict) else str(r)
    for r in leaked_ticket_output
]

# Mitigation: remove resolution_time_hours
ticket_agent_clean = Agent(
    skills=ClassificationSkill(
        name='priority_classification',
        input_template='Ticket: {ticket}',
        output_template='Priority: {predicted_priority}',
        labels={'predicted_priority': ["Urgent", "High", "Medium", "Low"]},
    ),
    environment=StaticEnvironment(
        df=ticket_training_leaked[['ticket', 'priority']],
        ground_truth_columns={'predicted_priority': 'priority'}
    )
)
ticket_agent_clean.learn()
clean_ticket_output = ticket_agent_clean.run(ticket_test[['ticket']])
clean_ticket_preds = [
    r.get('predicted_priority', 'Unknown') if isinstance(r, dict) else str(r)
    for r in clean_ticket_output
]

gt_ticket = ticket_test['priority'].tolist()
evaluate(gt_ticket, leaked_ticket_preds, "With feature leakage (resolution_time_hours)")
evaluate(gt_ticket, clean_ticket_preds,  "After mitigation (leaked feature removed)")

print("\n⚠  Diagnosis: Any feature computed after the label is assigned should be")
print("   excluded from the input template. Audit each column's data-collection time.")

# COMMAND ----------

# MAGIC %md
# ## Use Case 5: Data Contamination (Test-Train Overlap)
#
# **Scenario**: Due to a deduplication bug, several rows from the test set are also
# present in the training set. The agent effectively memorises the answers, inflating
# evaluation metrics.
#
# **Detection**: Exact-match check between train and test split reveals duplicates;
# performance drops significantly after deduplication.

# COMMAND ----------

print("\n" + "=" * 70)
print("USE CASE 5: DATA CONTAMINATION (TEST-TRAIN OVERLAP)")
print("=" * 70)

base_train = pd.DataFrame([
    {"review": "Amazing product! Exceeded all expectations.", "sentiment": "Positive"},
    {"review": "Terrible quality. Very disappointed.",         "sentiment": "Negative"},
    {"review": "It's okay, nothing special.",                  "sentiment": "Neutral"},
    {"review": "Love it! Highly recommend to everyone.",       "sentiment": "Positive"},
    {"review": "Waste of money. Complete disappointment.",     "sentiment": "Negative"},
    {"review": "Good product, but could be better.",           "sentiment": "Neutral"},
])

# Test set — 3 rows are exact duplicates from training (contamination)
contaminated_test = pd.DataFrame([
    {"review": "Amazing product! Exceeded all expectations.", "sentiment": "Positive"},  # duplicate
    {"review": "Terrible quality. Very disappointed.",        "sentiment": "Negative"},  # duplicate
    {"review": "Fantastic quality and fast delivery!",        "sentiment": "Positive"},
    {"review": "Defective product upon arrival.",             "sentiment": "Negative"},
    {"review": "It's okay, nothing special.",                 "sentiment": "Neutral"},   # duplicate
    {"review": "Okay product for the price.",                 "sentiment": "Neutral"},
])

# Detect contamination
overlap = pd.merge(base_train, contaminated_test, on='review', suffixes=('_train', '_test'))
print(f"\nContaminated rows found: {len(overlap)}")
print(overlap[['review']].to_string(index=False))

# Evaluate on contaminated test (inflated)
contaminated_agent = build_sentiment_agent(base_train)
contaminated_preds = run_agent(contaminated_agent, contaminated_test)
evaluate(contaminated_test['sentiment'].tolist(), contaminated_preds,
         "Evaluation on contaminated test set (inflated)")

# Mitigation: remove duplicates from test set
clean_test_dedup = contaminated_test[
    ~contaminated_test['review'].isin(base_train['review'])
].reset_index(drop=True)
print(f"\nClean test set size after deduplication: {len(clean_test_dedup)}")

clean_dedup_agent = build_sentiment_agent(base_train)
clean_dedup_preds = run_agent(clean_dedup_agent, clean_test_dedup)
evaluate(clean_test_dedup['sentiment'].tolist(), clean_dedup_preds,
         "After mitigation (duplicates removed from test set)")

print("\n⚠  Diagnosis: Always run an exact-match (and near-match) deduplication check")
print("   between train and test splits before finalising your evaluation set.")

# COMMAND ----------

# MAGIC %md
# ## Use Case 6: Pipeline Leakage (Preprocessing Fitted on Full Dataset)
#
# **Scenario**: A keyword-frequency feature is computed over the entire dataset
# (train + test combined) before splitting. High-frequency words in the test set
# influence what the agent treats as informative, giving an unfair signal.
#
# **Detection**: Recompute features using only training data statistics;
# evaluate again and observe the accuracy drop.

# COMMAND ----------

print("\n" + "=" * 70)
print("USE CASE 6: PIPELINE LEAKAGE (PREPROCESSING ON FULL DATASET)")
print("=" * 70)

# Full dataset before split
full_data = pd.DataFrame([
    {"review": "Excellent service, highly recommend!",        "sentiment": "Positive"},
    {"review": "Highly recommend to all my friends.",         "sentiment": "Positive"},
    {"review": "Worst product I've ever bought.",             "sentiment": "Negative"},
    {"review": "Broken on arrival. Absolute disaster.",       "sentiment": "Negative"},
    {"review": "Decent, nothing to write home about.",        "sentiment": "Neutral"},
    {"review": "Average quality, meets basic needs.",         "sentiment": "Neutral"},
    {"review": "Highly recommend — superb build quality!",   "sentiment": "Positive"},
    {"review": "I regret this purchase. Very poor quality.",  "sentiment": "Negative"},
    {"review": "Satisfactory. Does exactly what it says.",    "sentiment": "Neutral"},
    {"review": "Absolutely love it. Highly recommend!",       "sentiment": "Positive"},
])

# LEAKED: keyword flag computed on all 10 rows including test rows
all_reviews = full_data['review'].str.lower()
# "highly recommend" appears 4× — using global frequency leaks test-set signal
full_data['has_recommend'] = all_reviews.str.contains('highly recommend').astype(int)

train_leaked = full_data.iloc[:6].copy()
test_leaked  = full_data.iloc[6:].copy()

print("\n--- LEAKED: 'has_recommend' computed on full dataset ---")
print(full_data[['review', 'has_recommend', 'sentiment']].to_string(index=False))

# Build agent with the leaked feature in the input template
def build_agent_with_flag(df: pd.DataFrame) -> Agent:
    df = df.copy()
    df['review_with_flag'] = df.apply(
        lambda r: f"{r['review']} [recommend_flag={r['has_recommend']}]", axis=1
    )
    return Agent(
        skills=ClassificationSkill(
            name='sentiment_classification',
            input_template='Review: {review_with_flag}',
            output_template='Sentiment: {predicted_sentiment}',
            labels={'predicted_sentiment': ["Positive", "Negative", "Neutral"]},
        ),
        environment=StaticEnvironment(
            df=df,
            ground_truth_columns={'predicted_sentiment': 'sentiment'}
        )
    )

leaked_pipeline_agent = build_agent_with_flag(train_leaked)
leaked_pipeline_agent.learn()
leaked_pipeline_output = leaked_pipeline_agent.run(test_leaked)
leaked_pipeline_preds = [
    r.get('predicted_sentiment', 'Unknown') if isinstance(r, dict) else str(r)
    for r in leaked_pipeline_output
]

# Mitigation: compute keyword flag using only training data statistics
train_clean = full_data.iloc[:6].copy()
test_clean  = full_data.iloc[6:].copy()
train_reviews_lower = train_clean['review'].str.lower()
train_clean['has_recommend'] = train_reviews_lower.str.contains('highly recommend').astype(int)
# Apply training vocabulary to test set independently
test_clean['has_recommend'] = test_clean['review'].str.lower().str.contains('highly recommend').astype(int)

clean_pipeline_agent = build_agent_with_flag(train_clean)
clean_pipeline_agent.learn()
clean_pipeline_output = clean_pipeline_agent.run(test_clean)
clean_pipeline_preds = [
    r.get('predicted_sentiment', 'Unknown') if isinstance(r, dict) else str(r)
    for r in clean_pipeline_output
]

gt_pipeline = test_leaked['sentiment'].tolist()
evaluate(gt_pipeline, leaked_pipeline_preds, "With pipeline leakage (flag from full dataset)")
evaluate(gt_pipeline, clean_pipeline_preds,  "After mitigation (flag from train-only statistics)")

print("\n⚠  Diagnosis: Any preprocessing step that aggregates statistics (frequency,")
print("   TF-IDF, normalisation, encoding) must be fit on training data only,")
print("   then applied independently to the test set.")

# COMMAND ----------

# MAGIC %md
# ## Summary: Leak Detection Checklist

print("\n" + "=" * 70)
print("LEAK DETECTION CHECKLIST")
print("=" * 70)

checklist = [
    ("Ground Truth Leakage",
     "Inspect input templates — label keywords must not appear in any input field."),
    ("Temporal Leakage",
     "Audit every column's collection timestamp vs. the prediction window."),
    ("Annotator Bias Leakage",
     "Compute Cohen's kappa across annotators; adjudicate disagreements before training."),
    ("Feature Leakage",
     "Trace each feature's data lineage — exclude anything computed after the label."),
    ("Data Contamination",
     "Run exact-match + fuzzy deduplication between train and test before evaluation."),
    ("Pipeline Leakage",
     "Fit all preprocessing (scalers, encoders, frequency counts) on training data only."),
]

for i, (name, tip) in enumerate(checklist, 1):
    print(f"\n{i}. {name}")
    print(f"   → {tip}")

print("\n" + "=" * 70)
print("✓ All use cases completed")
print("=" * 70)
