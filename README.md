# FlipItNews — News Category Classification (NLP Project)

## Status: Complete

Built and delivered a full Jupyter notebook (`FlipItNews_Classification.ipynb`, 46 cells, executed with no errors) plus an 18-page PDF export (well under the 50-page submission cap), covering the full assignment brief end to end.

## Dataset
`flipitnews-data.csv` (provided directly by user upload, since the Google Drive link was blocked by the sandbox's egress policy). 2,225 articles, 2 columns (`Article`, `Category`), no missing values, 99 duplicate rows (left in place).

Category distribution: Sport 511, Business 510, Politics 417, Technology 401, Entertainment 386.

## Pipeline built
1. EDA — shape, category distribution (bar chart), article length distribution, null/duplicate checks.
2. `preprocess_text()` — user-defined function: remove non-letters (regex) → lowercase → tokenize (NLTK `word_tokenize`) → remove stopwords + stray single-char tokens → lemmatize (WordNet). Shown before/after on a sample article.
3. Label-encoded `Category` → `Category_Encoded` (Business=0, Entertainment=1, Politics=2, Sports=3, Technology=4).
4. `vectorize_text(corpus, method='bow'|'tfidf')` — user-selectable vectorization, `max_features=5000`. Notebook runs with `VECTORIZATION_METHOD = 'tfidf'`.
5. 75:25 stratified train-test split → train (1668, 5000), test (557, 5000).
6. Naive Bayes trained first as the baseline (simple approach) with confusion matrix + classification report.
7. `train_and_evaluate()` helper function used to train/evaluate Decision Tree, KNN, and Random Forest identically.
8. Model comparison table + bar chart, with written observations.

## Results (TF-IDF, 5000 features, 75:25 split, random_state=42)
| Model | Test Accuracy |
|---|---|
| Naive Bayes | 0.9767 |
| Random Forest (n_estimators=200) | 0.9695 |
| K-Nearest Neighbors (k=5) | 0.9461 |
| Decision Tree | 0.8600 |

Naive Bayes was the best performer; Decision Tree the weakest (single-tree overfitting on sparse high-dim text features).

## Questionnaire — answers given in the notebook
1. Total articles: 2,225
2. Most articles: Sport (511)
3. Technology articles: 401
4. Stop words definition/why removed — explained (noise reduction, dimensionality, non-discriminative).
5. Stemming vs Lemmatization — explained (rule-based crude vs vocabulary/morphology-aware).
6. TF-IDF judged more efficient/effective than plain BoW (down-weights common words, up-weights discriminative ones).
7. Train/test shapes: (1668, 5000) / (557, 5000).
8. Best performing model: (c) Naive Bayes.
9. Precision/recall equally important: True (content categorization use case, no asymmetric error cost) — reasoning given.

## Delivered files
Sent to user via chat: `FlipItNews_Classification.ipynb` and `FlipItNews_Classification.pdf`.
