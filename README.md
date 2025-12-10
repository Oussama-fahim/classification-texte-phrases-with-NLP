# 📚 NLP Text Classification with Keras & TensorFlow

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-API-red)

*A comprehensive Natural Language Processing pipeline for text classification using TensorFlow, Keras, and scikit-learn*

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Pipeline Overview](#-pipeline-overview) • [Code Explanation](#-detailed-code-explanation) • [Results](#-results)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Pipeline Overview](#-pipeline-overview)
- [Detailed Code Explanation](#-detailed-code-explanation)
- [Results & Performance](#-results--performance)
- [Advanced NLP Techniques](#-advanced-nlp-techniques)
- [Configuration & Hyperparameters](#-configuration--hyperparameters)
- [Use Cases](#-use-cases)
- [Troubleshooting](#-troubleshooting)
- [References](#-references)

---

## 🌟 Overview

This project demonstrates a complete **Natural Language Processing (NLP) pipeline** for text classification tasks. It covers everything from basic text tokenization to advanced deep learning models using TensorFlow and Keras. The notebook implements both traditional machine learning (Logistic Regression) and deep learning approaches (Neural Networks with Embeddings) on two different datasets.

### Key Learning Objectives:

- 📝 **Text Preprocessing**: Tokenization, padding, stopword removal
- 🔢 **Feature Engineering**: Converting text to numerical sequences
- 🤖 **Machine Learning**: Logistic Regression for text classification
- 🧠 **Deep Learning**: Keras Sequential models with Embedding layers
- 📊 **Model Evaluation**: Accuracy, precision, recall, F1-score
- 🎨 **Visualization**: Training history plots for deep learning models

---

## ✨ Features

### Core NLP Pipeline
- ✅ **Text Tokenization**: Convert text to integer sequences
- ✅ **Padding/Truncation**: Ensure uniform input length
- ✅ **Stopword Removal**: Filter out common English words
- ✅ **Out-of-Vocabulary Handling**: `<OOV>` token for unknown words
- ✅ **Label Encoding**: Convert categorical labels to numerical values

### Machine Learning Models
- 📊 **Logistic Regression**: Traditional ML approach
- 🧠 **Keras Sequential Model**: Deep learning with Embedding layers
- 🔄 **GlobalAveragePooling1D**: For dimensionality reduction
- 🎯 **Binary Classification**: Sarcasm detection and sentiment analysis

### Data Processing
- 🐼 **Pandas Integration**: DataFrame manipulation and storage
- 📥 **JSON Data Loading**: Handle structured text datasets
- 🧹 **Text Cleaning**: Lowercasing, punctuation removal
- 📈 **Data Splitting**: Train-test validation

### Visualization & Analysis
- 📉 **Training History Plots**: Loss and accuracy curves
- 🔍 **Vocabulary Analysis**: Word frequency and distribution
- 📋 **Model Summaries**: Layer-by-layer architecture details
- 📊 **Performance Metrics**: Comprehensive evaluation scores

---

## 📁 Project Structure

```
NLP-Text-Classification.ipynb
│
├── 📦 IMPORT LIBRARIES
│   ├── TensorFlow & Keras
│   ├── scikit-learn
│   ├── pandas & numpy
│   └── NLTK (Natural Language Toolkit)
│
├── 🎯 BASIC TOKENIZATION DEMO
│   ├── Simple sentence tokenization
│   ├── Word index creation
│   ├── Sequence padding (pre/post)
│   └── OOV token handling
│
├── 📚 LABELED DATASET CREATION
│   ├── Bilingual sentiment data (French/English)
│   ├── DataFrame conversion
│   ├── Advanced tokenization with OOV
│   └── Label encoding (bon→1, mauvais→0)
│
├── 🤖 MACHINE LEARNING CLASSIFICATION
│   ├── Logistic Regression implementation
│   ├── Train-test split (80-20)
│   └── Model training on tokenized sequences
│
├── 🎭 SARCASM DETECTION DATASET
│   ├── Dataset download from Google Drive
│   ├── JSON data loading and preprocessing
│   ├── Headline analysis
│   └── Column restructuring
│
├── 🧹 ADVANCED TEXT PREPROCESSING
│   ├── NLTK stopword installation
│   ├── English stopword definition
│   ├── Custom stopword removal function
│   └── Cleaned text storage
│
├── 🔢 ADVANCED TOKENIZATION PIPELINE
│   ├── Vocabulary building from cleaned text
│   ├── Sequence conversion
│   ├── Padding to fixed length (50 tokens)
│   └── NumPy array conversion for modeling
│
├── 📊 TRADITIONAL ML APPROACH
│   ├── Data standardization (StandardScaler)
│   ├── Logistic Regression training
│   ├── Model prediction
│   └── Performance metrics calculation
│
├── 🧠 DEEP LEARNING APPROACH
│   ├── Vocabulary size calculation
│   ├── Keras Sequential model architecture
│   ├── Embedding layer configuration
│   ├── Model compilation and training
│   ├── Model evaluation
│   └── Training history visualization
│
└── 📈 RESULTS & VISUALIZATION
    ├── Loss curves comparison
    ├── Accuracy progression
    └── Model performance summary
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or Jupyter Lab
- Google Colab (recommended for GPU acceleration)

### Step 1: Environment Setup

```bash
# Clone repository (if applicable)
git clone https://github.com/Oussamafahim/classification-texte-phrases-with-NLP.git
cd nlp-text-classification

# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
# Install core dependencies
pip install tensorflow pandas numpy scikit-learn matplotlib nltk

# For Jupyter support
pip install jupyter notebook

# Download NLTK data
python -c "import nltk; nltk.download('stopwords')"
```

### Step 3: Run Jupyter Notebook

```bash
jupyter notebook NLP.ipynb
```

### Google Colab Alternative

Simply upload the notebook to [Google Colab](https://colab.research.google.com/) - all dependencies are pre-installed!

---

## 🎯 Quick Start

### Run the Complete Pipeline

```python
# Execute all cells in sequence
# The notebook will:
# 1. Install required packages
# 2. Load and preprocess data
# 3. Train machine learning models
# 4. Train deep learning models
# 5. Evaluate and visualize results
```

### Test Basic Tokenization

```python
# Quick tokenization example
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = ['i love my dog', 'I, love my cat', 'you love my dog!']
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)

# View word index
print(tokenizer.word_index)
# Output: {'love': 1, 'my': 2, 'dog': 3, 'i': 4, 'cat': 5, 'you': 6}
```

---

## 📊 Pipeline Overview

### Complete NLP Workflow

```
Raw Text
    ↓
Text Cleaning
    ↓
Tokenization
    ↓
Sequence Padding
    ↓
Feature Extraction
    ↓
Model Training
    ↓
Evaluation
    ↓
Visualization
```

### Dataset Flow

```
Two Datasets:
1. Custom Sentiment Dataset (10 samples)
   - Purpose: Demonstration of basic pipeline
   - Labels: "bon" (good) vs "mauvais" (bad)
   
2. Sarcasm Detection Dataset (26,709 samples)
   - Source: News Headlines JSON
   - Purpose: Real-world binary classification
   - Labels: 0 (not sarcastic) vs 1 (sarcastic)
```

---

## 📝 Detailed Code Explanation

### Cell 1-2: Import Libraries

```python
import tensorflow as tf
from tensorflow import keras
# TensorFlow provides the backend for deep learning operations
# Keras offers high-level neural networks API
```

**Purpose**: Import core deep learning frameworks for building and training neural networks.

### Cell 3: Basic Tokenization Example

```python
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'i love my dog',
    'I, love my cat',
    'you love my dog!',
    'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
```
**Key Concepts**:
- `Tokenizer`: Converts text to sequences of integers
- `num_words=100`: Keeps only top 100 most frequent words
- `fit_on_texts()`: Learns vocabulary from sentences
- Word index maps each unique word to an integer

### Cell 4: Sequence Padding

```python
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=5)
```
**Padding Types**:
- **Pre-padding**: Default (adds zeros at beginning)
- **Post-padding**: `padding='post'` (adds zeros at end)
- **Truncation**: `truncating='post'` (cuts from end if too long)

### Cell 5-6: Labeled Dataset Creation

```python
labeled_sentences = {
    "sentences": ["This movie was excellent", ...],
    "lebel": ["bon", "mauvais", ...]  # Note: French labels
}

df = pd.DataFrame(labeled_sentences)
```
**Features**:
- Bilingual sentiment analysis (French labels)
- Balanced dataset (5 positive, 5 negative)
- Demonstrates DataFrame integration

### Cell 7-11: Advanced Tokenization Pipeline

```python
# Initialize with OOV token
tokenizer_classifier = Tokenizer(num_words=100, oov_token="<OOV>")

# Fit on sentences
tokenizer_classifier.fit_on_texts(df['sentences'])

# Convert to sequences
sequences_classifier = tokenizer_classifier.texts_to_sequences(df['sentences'])

# Apply padding (post-padding to length 6)
padded_classifier = pad_sequences(sequences_classifier, maxlen=6, padding='post')

# Label encoding
df['label'] = df['lebel'].map({'bon': 1, 'mauvais': 0})
```

**OOV Token Importance**: 
- `<OOV>` handles unknown words during prediction
- Prevents errors when encountering new vocabulary
- Assigned index 1 in word index

### Cell 12: Logistic Regression Training

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df['tokenized_sentences'], 
    df['label'], 
    test_size=0.2, 
    random_state=42
)

model = LogisticRegression()
history = model.fit(X_train.tolist(), y_train)
```

**Train-Test Split**:
- 80% training, 20% testing
- `random_state=42` ensures reproducibility
- Converts tokenized sequences to list format

### Cell 13-16: Sarcasm Dataset Loading

```python
# Download dataset from Google Drive
!gdown --id 1xRU3xY5-tkiPGvlz5xBJ18_pHWSRzI4v

# Load JSON data
df = pd.read_json('/content/sarcasm.json')

# Remove unnecessary columns
df.drop(columns='article_link', inplace=True)
```

**Dataset Characteristics**:
- 26,709 news headlines
- Binary labels: 0 (serious), 1 (sarcastic)
- Source: The Onion (sarcastic) vs Huffington Post (serious)

### Cell 17-20: Advanced Text Preprocessing

```python
# Download NLTK stopwords
nltk.download('stopwords')

# Define English stopwords
english_stopwords = stopwords.words('english')

# Custom stopword removal function
def remove_stopwords(text):
    words = text.lower().split()
    filtered_words = [word for word in words if word not in english_stopwords]
    return " ".join(filtered_words)

# Apply to dataset
df['cleaned_headline'] = df['headline'].apply(remove_stopwords)
```

**Stopword Removal Benefits**:
- Reduces vocabulary size by ~30%
- Removes non-informative words (the, is, at, etc.)
- Improves model focus on meaningful content words
- Lowers computational complexity

### Cell 21-24: Advanced Tokenization for Sarcasm Dataset

```python
# Create tokenizer with OOV handling
tokenizer_cleaned = Tokenizer(oov_token="<OOV>")
tokenizer_cleaned.fit_on_texts(df['cleaned_headline'])

# Vocabulary statistics
vocab_size = len(tokenizer_cleaned.word_index) + 1
print(f"Found {vocab_size} unique tokens.")

# Convert to sequences
sequences_cleaned = tokenizer_cleaned.texts_to_sequences(df['cleaned_headline'])
```

**Vocabulary Analysis**:
- 29,571 unique tokens in cleaned headlines
- Most common: "new", "trump", "man", "one", "report"
- `<OOV>` token handles unseen words during inference

### Cell 25-29: Logistic Regression on Sarcasm Data

```python
# Data standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
model_logistic = LogisticRegression(max_iter=1000)
model_logistic.fit(X_train_scaled, y_train)

# Prediction and evaluation
y_pred = model_logistic.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
```

**Performance Metrics**:
- **Accuracy**: Proportion of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Cell 30-34: Deep Learning Model with Keras

```python
# Model architecture
model = Sequential([
    Embedding(vocab_size, 16),  # Convert words to dense vectors
    GlobalAveragePooling1D(),    # Reduce sequence dimension
    Dense(24, activation='relu'), # Hidden layer
    Dense(1, activation='sigmoid') # Output layer (binary classification)
])

# Model compilation
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Model training
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)
```

**Layer Explanation**:
1. **Embedding Layer**: 
   - Input: Integer sequences
   - Output: Dense word vectors (16 dimensions)
   - Learns semantic relationships between words

2. **GlobalAveragePooling1D**:
   - Averages across sequence dimension
   - Reduces computational complexity
   - Creates fixed-length output regardless of input length

3. **Dense Layers**:
   - Learn hierarchical feature representations
   - ReLU activation for non-linearity
   - Sigmoid activation for binary probability output

### Cell 35: Training History Visualization

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss curve
axes[0].plot(history.history['loss'], label='Train Loss', marker='o')
axes[0].plot(history.history['val_loss'], label='Val Loss', marker='s')

# Accuracy curve
axes[1].plot(history.history['accuracy'], label='Train Accuracy', marker='o')
axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', marker='s')
```

**Visualization Insights**:
- **Loss Curves**: Monitor overfitting (train vs validation gap)
- **Accuracy Curves**: Track learning progress
- **Early Stopping**: Can be implemented when validation loss plateaus

---

## 📈 Results & Performance

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| Logistic Regression | 57.64% | 57.59% | 13.43% | 21.78% | ~5 seconds |
| Keras Sequential | 79.82% | N/A | N/A | N/A | ~52 seconds |

### Key Observations

1. **Deep Learning Superiority**:
   - Keras model achieves ~22% higher accuracy than Logistic Regression
   - Better at capturing complex patterns in text data
   - Handles sequential information through embeddings

2. **Logistic Regression Limitations**:
   - Poor recall (13.43%) indicates difficulty identifying sarcastic headlines
   - Treats text as bag-of-words without sequence consideration
   - Limited capacity for complex feature interactions

3. **Training Dynamics**:
   - Keras model shows steady improvement over 10 epochs
   - Validation accuracy plateaus around epoch 4-5
   - Minimal overfitting observed (train/val curves close)

### Performance Metrics Deep Dive

#### Logistic Regression Challenges:
- **Low Recall**: Model misses many sarcastic headlines
- **Class Imbalance**: Dataset might have uneven sarcasm distribution
- **Feature Representation**: Bag-of-words approach loses word order information

#### Keras Model Advantages:
- **Embedding Learning**: Captures semantic relationships
- **Sequence Awareness**: Maintains word order through processing
- **Non-linear Transformations**: Multiple dense layers learn complex patterns

---

## 🚀 Advanced NLP Techniques

### 1. Embedding Layer Analysis

```python
# Access learned embeddings
embedding_layer = model.layers[0]
weights = embedding_layer.get_weights()[0]

# Find similar words using cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

def find_similar_words(word, word_index, weights, top_n=5):
    if word not in word_index:
        return []
    word_idx = word_index[word]
    word_vec = weights[word_idx].reshape(1, -1)
    similarities = cosine_similarity(word_vec, weights)[0]
    similar_indices = similarities.argsort()[-top_n-1:-1][::-1]
    return [(list(word_index.keys())[i], similarities[i]) for i in similar_indices]
```

### 2. Hyperparameter Tuning

```python
# Grid search for optimal parameters
param_grid = {
    'vocab_size': [10000, 20000, 30000],
    'embedding_dim': [16, 32, 64],
    'hidden_units': [16, 32, 64],
    'learning_rate': [0.001, 0.01, 0.1]
}

# Implementation using Keras Tuner
import kerastuner as kt

def build_model(hp):
    model = Sequential()
    model.add(Embedding(
        input_dim=hp.Int('vocab_size', 10000, 30000, step=5000),
        output_dim=hp.Int('embedding_dim', 16, 64, step=16)
    ))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(
        units=hp.Int('hidden_units', 16, 64, step=16),
        activation='relu'
    ))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice('learning_rate', [0.001, 0.01, 0.1])
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
```

### 3. Advanced Architectures

```python
# LSTM Model for better sequence understanding
model_lstm = Sequential([
    Embedding(vocab_size, 64),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Bidirectional LSTM
model_bilstm = Sequential([
    Embedding(vocab_size, 64),
    Bidirectional(LSTM(64)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# CNN for text classification
model_cnn = Sequential([
    Embedding(vocab_size, 64, input_length=50),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

---

## ⚙️ Configuration & Hyperparameters

### Tokenization Settings

```python
# Optimal tokenization configuration
TOKENIZER_CONFIG = {
    'num_words': 20000,           # Vocabulary size
    'oov_token': '<OOV>',         # Out-of-vocabulary token
    'filters': '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',  # Characters to filter
    'lower': True,                # Convert to lowercase
    'split': ' ',                 # Word separator
    'char_level': False           # Character-level tokenization
}
```

### Model Hyperparameters

```python
# Keras Model Parameters
MODEL_PARAMS = {
    'embedding_dim': 64,          # Word vector dimension
    'hidden_units': 128,          # Dense layer neurons
    'dropout_rate': 0.5,          # Regularization
    'learning_rate': 0.001,       # Adam optimizer
    'batch_size': 32,             # Training batch size
    'epochs': 20,                 # Training iterations
    'validation_split': 0.2,      # Validation data proportion
    'patience': 5                 # Early stopping patience
}
```

### Training Configuration

```python
# Callbacks for improved training
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.00001
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    )
]
```

---

## 🎯 Use Cases

### 1. Sentiment Analysis
```python
# Adapt for product reviews
reviews = ["Great product!", "Terrible experience", ...]
# Convert to: Positive (1) / Negative (0)
```

### 2. Spam Detection
```python
# Classify emails/messages
messages = ["Win free money!", "Meeting reminder", ...]
# Convert to: Spam (1) / Not Spam (0)
```

### 3. Topic Classification
```python
# Multi-class classification
articles = ["Sports news", "Technology update", "Political analysis", ...]
# Convert to: [Sports, Tech, Politics, ...]
```

### 4. Intent Recognition
```python
# Chatbot applications
user_queries = ["What's the weather?", "Book a flight", ...]
# Convert to: [weather_query, booking_request, ...]
```

---

## 🔧 Troubleshooting

### Common Issues & Solutions

#### Issue 1: Memory Error with Large Vocabulary
```python
# Solution: Limit vocabulary size
tokenizer = Tokenizer(num_words=20000)  # Instead of keeping all words

# Alternative: Use hashing trick
from sklearn.feature_extraction.text import HashingVectorizer
vectorizer = HashingVectorizer(n_features=10000)
```

#### Issue 2: Overfitting in Deep Learning Model
```python
# Solution: Add regularization
model = Sequential([
    Embedding(vocab_size, 64),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),  # 50% dropout
    Dense(1, activation='sigmoid')
])
```

#### Issue 3: Class Imbalance
```python
# Solution: Class weights
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

# Use in model.fit()
model.fit(..., class_weight=class_weight_dict)
```

#### Issue 4: Slow Training
```python
# Solutions:
# 1. Reduce batch size
batch_size = 64  # Instead of 32

# 2. Use mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# 3. Implement gradient accumulation
# (Simulate larger batch sizes without memory increase)
```

#### Issue 5: Poor Validation Performance
```python
# Solutions:
# 1. Increase model capacity
model.add(Dense(256, activation='relu'))

# 2. Use pre-trained embeddings
embedding_layer = tf.keras.layers.Embedding(
    vocab_size,
    300,
    embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
    trainable=False  # Freeze during training
)

# 3. Data augmentation
# Back-translation, synonym replacement, etc.
```

---

## 📚 References

### Academic Papers
1. **Attention Is All You Need** - Vaswani et al. (2017)
2. **BERT: Pre-training of Deep Bidirectional Transformers** - Devlin et al. (2018)
3. **GloVe: Global Vectors for Word Representation** - Pennington et al. (2014)
4. **Universal Language Model Fine-tuning for Text Classification** - Howard & Ruder (2018)

### Books & Courses
- **Deep Learning with Python** - François Chollet
- **Natural Language Processing in Action** - Hobson Lane, Cole Howard, Hannes Hapke
- **Speech and Language Processing** - Daniel Jurafsky & James H. Martin
- **Coursera NLP Specialization** - deeplearning.ai

### Useful Resources
- [TensorFlow Text Classification Guide](https://www.tensorflow.org/tutorials/keras/text_classification)
- [Keras Preprocessing Documentation](https://keras.io/api/preprocessing/text/)
- [scikit-learn Text Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [NLTK Official Documentation](https://www.nltk.org/)

---

## 🤝 Contributing

### Areas for Improvement
- 📈 **Transformer Models**: Implement BERT or RoBERTa
- 🌐 **Multilingual Support**: Extend to non-English text
- 🔍 **Explainable AI**: Add SHAP/LIME for model interpretability
- ⚡ **Optimization**: Implement quantization and pruning
- 🚀 **Deployment**: Create Flask/FastAPI serving endpoint

### Contribution Guidelines
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code follows PEP 8 standards
5. Submit a pull request with detailed description

---

## 📞 Support & Contact

For questions, issues, or collaborations:
- 📧 Email: Oussamafahim2017@gmail.com
- 📱 Phone: +212645468306

---

<div align="center">

## 🎓 Educational Value

This notebook serves as an excellent resource for:
- **University Students**: Learning NLP fundamentals
- **Data Scientists**: Building production-ready text classification pipelines
- **Researchers**: Experimenting with different architectures
- **Educators**: Teaching machine learning and deep learning concepts

**⭐ If you found this project helpful, please consider giving it a star! ⭐**

*Made with ❤️ by Oussama Fahim*

</div>
