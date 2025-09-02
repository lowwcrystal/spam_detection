# 📱 SMS Spam Detection

A machine learning project to classify SMS messages as real or spam using through both traditional NLP (TF-IDF + linear models) and transformer embeddings, such as BART.

Requirements
- Python 3.9+
- Install dependencies:
python -m venv spamvenv
source spamvenv/bin/activate or Windows: spamvenv\Scripts\activate
pip install
torch
transformers
tqdm
scikit-learn
pandas
numpy
matplotlib
jupyter

# Results
**Evaluating on the training data:**
Accuracy: 0.9964109659194946
Sensitivity: 0.9763912558555603
Specificity: 0.9994825124740601
Precision: 0.9965576529502869
**Evaluating on the validation data:**
Accuracy: 0.9874326586723328
Sensitivity: 0.9220778942108154
Specificity: 0.9979166388511658
Precision: 0.9861111044883728
