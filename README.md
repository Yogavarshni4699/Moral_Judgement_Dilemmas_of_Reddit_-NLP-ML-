# Moral_Judgement_Dilemmas_of_Reddit (NLP & ML)
Leveraging natural language processing and Machine learning techniques, to analyze the textual content of posts and deliver an immediate judgment of "Yes" or "No" to the poster, simulating a virtual community response. 

Youtube link: https://www.youtube.com/watch?v=aXJtnJFKUwM

1. Project Rationale & Problem Statement

The Problem

Context: Reddit’s r/AmItheJerk (similar to r/AmItheAsshole) is a popular community where people post personal dilemmas, seeking moral judgments from commenters. These judgments typically label the original poster (OP) as either “Yes, The Jerk” or “No, Not The Jerk.”

Challenge: Human-driven judgments rely on readers’ availability. There can be delays, especially for posts requiring quick moral feedback.

Goal: Automate these judgments using machine learning (ML) and natural language processing (NLP) so that the OP and other readers can get near-instant feedback, reflecting the community consensus.

Objectives
Data Acquisition: Collect posts and comments from r/AmItheJerk, focusing on top-level comments where people explicitly give their verdict.
Model Development: Build and compare multiple ML models—Random Forest, Logistic Regression, Decision Tree, Support Vector Classifier (SVC), and XGBoost—to classify a post as YTJ or NTJ.
Deployment: Provide a user-facing web app (via Streamlit) where someone can paste the text of a Reddit post and get an automated classification.

2. Data Collection & Extraction
Source & Format
Pushshift.io: You pulled monthly data dumps of Reddit submissions and comments.
Date Range: December 2023 – April 2024 for the r/AmItheJerk subreddit.
Raw Size:
Submissions (Posts): ~125GB
Comments: ~250GB
Conversion: The data was initially in compressed .zst block files. You converted them to CSV format, significantly reducing file size (e.g., 235.2 MB for posts and 4.5 GB for comments).
Data Extraction Pipeline
Raw Data → .zst Files: Downloaded from pushshift.io in monthly chunks.
.zst Files → CSV: Decompressed and merged into a final CSV for submissions and a final CSV for comments.

3. Data Preprocessing
System Architecture Overview
Your architecture (as shown in the slides) has these key stages:

Data Collection: Gathering submissions and comments from the pushshift.io dumps.
Preprocessing: Cleaning and filtering relevant columns.
Labeling: Assigning YTJ/NTJ labels.
Feature Engineering: Using NLP steps like TF-IDF.
Modeling: Training and tuning multiple ML models.
Evaluation: Checking accuracy, precision, recall, and F1 scores.
Deployment: Building a Streamlit web application for predictions.
Cleaning & Merging

Comments:

Retain only link_id, parent_id, and body.
Filter out rows where body is [removed], [deleted], or NaN.
Keep only top-level comments (i.e., comments whose parent_id matches the post’s link_id).

Posts (Submissions):

Retain selftext and name (ID column).
Merge with the comments data on matching IDs so each post can be linked to its top-level comments.

Data Labeling
Identifying Judgment Tags: You searched the comment text for tags like yta, esh, nta, etc.
Assigning Classes:
NTJ (Not The Jerk): “nta,” “nah,” “ywnbta,” “yntah,” etc.
YTJ (You’re The Jerk): “yta,” “esh,” “ywbta,” etc.
Majority Voting: If multiple top-level comments exist for a single post, whichever label (NTJ or YTJ) appears most frequently becomes the final label.

Resulting Dataset
Final shape: A single dataset where each row corresponds to a Reddit post, combined with the assigned YTJ or NTJ label.
Columns typically include link_id, label, selftext, and name.

4. Exploratory Data Analysis (EDA)
Although your main target is classification, you also examined:

Distribution of Word/Character Counts: Showed that many posts/comments are relatively short, but a few are very long.
Most Frequent Words: Common English words (like “like,” “would,” “know,” “people”) appear often, highlighting the need for stop word removal.
Author & Score Distribution: Confirmed a skew (a few high-engagement posts vs. many with lower engagement).
This step helped you confirm which columns were valuable and guided the text cleaning approach (lemmatization, stop word removal).

5. Feature Engineering

Text Processing
Stop Word Removal: Eliminates common words (e.g., “the,” “and,” “or”) to reduce noise.
Lemmatization: Normalizes words to their base forms (e.g., “running,” “ran,” and “runs” → “run”).

TF-IDF Vectorization
TF (Term Frequency): Counts how often a word appears in a document.
IDF (Inverse Document Frequency): Down-weights words that appear across many documents.

Result: A numerical matrix representing text features, ready for ML models.

6. Modeling Phase
Data Splitting
You divided the labeled data into Training (70%), Validation (15%), and Test (15%) sets using stratified sampling to preserve the YTJ/NTJ ratio.

Algorithms Explored

1. Random Forest

Ensemble method using multiple decision trees.
Good at handling large feature spaces (like text data).
Tends to produce strong baseline results.

2. Logistic Regression

Classic linear model, often effective in text classification tasks with sparse data.
Simple and computationally efficient.

3. Decision Tree

Simple to interpret but prone to overfitting.
Good for a quick baseline and for interpretability checks.

4. Support Vector Classifier (SVC)

Can handle high-dimensional data well.
Sensitive to parameter choices (e.g., kernel type, regularization C).

5. XGBoost

Gradient boosting method known for high performance.
Often outperforms simpler models when tuned properly.

Modeling Scenarios
Baseline: Using default parameters to get an initial benchmark.

Hyperparameter Tuning (GridSearchCV):
You systematically search for the best combination of parameters (e.g., max_depth, n_estimators, C, kernel, etc.).
3-Fold Cross-Validation ensures more robust selection of optimal hyperparameters.
SMOTE + TSVD (Optional):
SMOTE: Synthetic Minority Oversampling Technique to address class imbalance (if “Yes” is less frequent than “No”).
TSVD: Truncated Singular Value Decomposition to reduce dimensionality from the high-dimensional TF-IDF vectors.

7. Evaluation & Classification Reports
Below are highlights from the classification reports shown in your slides, focusing on precision, recall, f1-score, and accuracy for both Test and Validation sets.

1. Random Forest
Best Parameters: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100} (example from the slide).
Accuracy: ~0.72 on both Test and Validation.
Precision/Recall:
Class 0 (“NTJ”) has high recall (close to 1.00), meaning it rarely misses “No” examples.
Class 1 (“YTJ”) has lower recall (~0.02 on test), indicating the model struggles to detect “Yes” posts accurately.
F1-Score: Weighted average around 0.62–0.73, reflecting the imbalance in labeling “Yes” vs. “No.”

2. Logistic Regression
Best Parameters: {'C': 0.01, 'class_weight': 'balanced', 'max_iter': 100, 'penalty': 'l1', 'solver': 'liblinear'}
Accuracy: ~0.72 on test.
Precision/Recall: Class 1 is particularly difficult for logistic regression to capture, often with near 0 recall in some scenarios.
Implication: The linear decision boundary struggles with subtle textual cues for “Yes” judgments.

3. Decision Tree
Best Parameters: {'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 2}
Accuracy: ~0.53–0.60, lower than the ensemble methods or logistic regression.
Precision/Recall: Balanced but relatively low, indicating the model is simpler and can over/underfit easily.

4. Support Vector Classifier
Best Parameters: {'C': 1, 'kernel': 'rbf'}
Accuracy: ~0.72–0.75 on the test set, a solid performer.
Precision/Recall:
Class 0 recall is often high, but class 1 recall remains low.
Weighted F1 can be ~0.60–0.70, depending on the exact data split.

5. XGBoost
Best Parameters: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100}
Accuracy: ~0.69–0.74 across different splits.
Precision/Recall: Similar pattern: class 0 is easy to catch, class 1 is missed more often.
Insights: XGBoost can outperform simpler models but still struggles with minority classes if the imbalance is stark.

Overall Observation:

NTJ (class 0) is almost always identified correctly.
YTJ (class 1) is harder to classify, often leading to lower recall for that class. This indicates the models see “Yes, The Jerk” examples as rarer or more context-specific, which is consistent with class imbalance and the subtlety of moral judgments.

8. Model Deployment with Streamlit

Saving the Models:
Each trained model (plus the TF-IDF transformations and label encoders) is stored via joblib so you can load them quickly in your web app.

Building the App:
Streamlit is used to create an interactive UI.
Users can type or paste a snippet of text (like a Reddit post’s body).
The app preprocesses the text (using the same TF-IDF pipeline) and runs predictions on all your chosen models.
It outputs whether the text is likely “YTJ” or “NTJ” along with confidence scores (or probabilities).

End-User Benefit
The OP or a casual reader can immediately see a system-generated verdict, mimicking community consensus without waiting for actual comments.

9. Conclusion & Future Directions

Conclusion:

You’ve successfully demonstrated how ML and NLP can replicate moral judgments on Reddit.
Despite high accuracy for “No” judgments, the models often miss “Yes” judgments due to data imbalance and the nuanced language of moral dilemmas.

Future Work:

Advanced NLP Models:
Incorporate BERT or GPT for better context understanding and nuanced language interpretation.
Deep Learning Architectures:
Try RNNs or Transformers to capture sequential patterns and subtle clues in text.
Real-time Feedback Loop:
Deploy the model in a live environment where user interactions can continually retrain or refine the model.
Class Imbalance Techniques:
Further refine SMOTE or investigate other sampling/weighting strategies to capture the “Yes” class more effectively.

