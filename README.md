# 💰 Expense Category Prediction API

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![Accuracy](https://img.shields.io/badge/Accuracy-93.33%25-brightgreen.svg)](https://github.com/ishaant97/Expense-Category-Prediction)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-accuracy deep learning model and REST API for automatically categorizing expense descriptions into 8 predefined categories. Built with TensorFlow, scikit-learn, and Flask.

![Demo](https://img.shields.io/badge/Demo-Available-blue.svg)

## ✨ Features

- 🎯 **93.33% Accuracy** - Achieved through data augmentation and optimized architecture
- 🚀 **REST API** - Easy integration with Flask-based endpoints
- 🔄 **Batch Processing** - Predict multiple expenses in a single request
- 🎲 **Confidence Scoring** - Returns uncertainty metrics for each prediction
- 📊 **8 Categories** - Food, Transport, Rent, Utilities, Entertainment, Groceries, Health, Education
- 🛡️ **Smart Fallback** - "Miscellaneous" category for low-confidence predictions
- 🔧 **Configurable** - Adjustable confidence thresholds via API

## � Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Model Performance](#-model-performance)
- [How It Works](#-how-it-works)
- [Usage Examples](#-usage-examples)
- [Training](#-training-your-own-model)
- [Configuration](#-configuration)
- [Contributing](#-contributing)
- [License](#-license)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip
- Virtual environment (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/ishaant97/Expense-Category-Prediction.git
cd Expense-Category-Prediction

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Running the API

```bash
cd src
python serve_model.py
```

The API will be available at `http://localhost:5000`

### Quick Test

```bash
# In a new terminal
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "pizza from dominos"}'
```

**Response:**
```json
{
  "success": true,
  "text": "pizza from dominos",
  "predicted_category": "Food",
  "confidence": 0.9523,
  "is_uncertain": false
}
```

## 📡 API Documentation

### Base URL
```
http://localhost:5000
```

### Endpoints

#### 1. Predict Single Expense
```http
POST /predict
Content-Type: application/json

{
  "text": "expense description",
  "include_all_predictions": false  // optional
}
```

#### 2. Batch Predict
```http
POST /batch_predict
Content-Type: application/json

{
  "texts": ["expense 1", "expense 2", ...]
}
```

#### 3. Get Configuration
```http
GET /config
```

#### 4. Update Configuration
```http
POST /config
Content-Type: application/json

{
  "confidence_threshold": 0.5
}
```

#### 5. Health Check
```http
GET /health
```

📚 **Full API Documentation:** [API_DOCUMENTATION.md](API_DOCUMENTATION.md)

## 📁 Project Structure

```
Expense-Category-Prediction/
├── data/
│   ├── expenses.csv                 # Original dataset (61 samples)
│   └── expenses_augmented.csv       # Augmented dataset (221 samples)
│
├── src/
│   ├── serve_model.py              # 🚀 Flask REST API
│   ├── train_with_augmented_data.py # 🎯 Main training script
│   ├── augment_data.py             # 📈 Data augmentation
│   ├── test_predictions.py         # 🧪 Interactive testing
│   ├── test_api.py                 # 🔧 API test suite
│   ├── compare_models.py           # 📊 Model comparison
│   └── check_data.py               # 📋 Dataset statistics
│
├── models/                         # 💾 Trained models
│   ├── expense_mlp_best.keras
│   ├── label_encoder_best.joblib
│   ├── vectorizer_best.joblib
│   └── model_config.json
│
├── requirements.txt
├── README.md
├── API_DOCUMENTATION.md            # 📖 Complete API reference
├── QUICK_START.md                  # ⚡ Quick setup guide
└── SUMMARY.md                      # 📝 Detailed improvements
```

## � Model Performance

### Accuracy Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **93.33%** |
| **Mean Confidence** | 91.9% |
| **Training Samples** | 221 (augmented from 61) |
| **Features** | 110 TF-IDF features |
| **Training Epochs** | 66 (early stopping) |

### Per-Category Accuracy

| Category | Precision | Recall | F1-Score | Samples |
|----------|-----------|--------|----------|---------|
| Education | 0.71 | 1.00 | 0.83 | 5 |
| Entertainment | 1.00 | 0.83 | 0.91 | 6 |
| **Food** | **1.00** | **1.00** | **1.00** | 6 |
| **Groceries** | **1.00** | **1.00** | **1.00** | 5 |
| **Health** | 0.86 | **1.00** | 0.92 | 6 |
| **Rent** | **1.00** | **1.00** | **1.00** | 5 |
| Transport | 1.00 | 0.67 | 0.80 | 6 |
| **Utilities** | **1.00** | **1.00** | **1.00** | 6 |

### Improvement Over Baseline

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Dataset Size | 61 | 221 | +262% |
| Test Accuracy | 40-60% | **93.33%** | **+55%** |
| Mean Confidence | 30-50% | 91.9% | **+70%** |
| TF-IDF Features | 1000 | 110 | Optimized ✅ |

## 🧠 How It Works

### 1. Data Augmentation
```python
# Expands limited dataset with synthetic examples
Original: "pizza from dominos" (61 samples)
Generated: 
  - "ordered pizza online"
  - "pizza at restaurant"  
  - "lunch at pizzeria"
  ... (221 samples total)
```

### 2. Smart Preprocessing
```python
# Preserves category-indicating keywords
IMPORTANT_WORDS = {'uber', 'rent', 'gym', 'movie', 'food', ...}

"uber to office" → "uber office" ✅  # Keeps "uber"
# vs old approach
"uber to office" → "office" ❌       # Lost context!
```

### 3. Optimized Architecture
```
Input (110 features)
    ↓
Dense(256) + BatchNorm + Dropout(0.5)
    ↓
Dense(128) + BatchNorm + Dropout(0.4)
    ↓
Dense(64) + Dropout(0.3)
    ↓
Output(8 categories) - Softmax
```

### 4. Confidence Threshold
```python
if confidence < 0.5:
    return "Miscellaneous"  # Prevents wrong guesses
else:
    return predicted_category
```

## 💻 Usage Examples

### Python Integration

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:5000/predict",
    json={"text": "coffee at starbucks"}
)
result = response.json()
print(f"{result['predicted_category']}: {result['confidence']:.2f}")
# Output: Food: 0.95

# Batch prediction
response = requests.post(
    "http://localhost:5000/batch_predict",
    json={
        "texts": [
            "pizza delivery",
            "uber ride",
            "monthly rent",
            "gym membership"
        ]
    }
)
predictions = response.json()['predictions']
for pred in predictions:
    print(f"{pred['text']} → {pred['predicted_category']}")
```

### JavaScript Integration

```javascript
async function categorizeExpense(text) {
  const response = await fetch('http://localhost:5000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });
  
  const result = await response.json();
  console.log(`${result.predicted_category} (${result.confidence})`);
  return result;
}

// Usage
categorizeExpense("netflix subscription");
```

### cURL

```bash
# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "electricity bill payment"}'

# Batch prediction
curl -X POST http://localhost:5000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["pizza", "uber", "rent", "gym"]}'
```

## 🎓 Training Your Own Model

### Option 1: Use Pre-trained Model (Recommended)
The repository includes a pre-trained model with 93.33% accuracy. Simply run the API:
```bash
cd src
python serve_model.py
```

### Option 2: Train from Scratch

```bash
cd src

# Train with automatic data augmentation
python train_with_augmented_data.py
```

**Training Process:**
1. ✅ Loads original dataset (61 samples)
2. ✅ Generates augmented samples (→ 221 samples)
3. ✅ Applies smart preprocessing
4. ✅ Trains optimized MLP model
5. ✅ Saves model artifacts to `models/`

**Expected Output:**
```
🎯 TEST ACCURACY: 0.9333 (93.33%)
📈 Training Summary:
  Best val accuracy: 0.9556
  Epochs trained: 66
🎲 Mean Confidence: 0.919
```

### Option 3: Add Your Own Data

1. Add expenses to `data/expenses.csv`:
```csv
description,category
your expense here,Category
another expense,Category
```

2. Retrain:
```bash
python train_with_augmented_data.py
```

## ⚙️ Configuration

### Model Configuration

Edit `models/model_config.json`:
```json
{
  "confidence_threshold": 0.5,
  "max_features": 110,
  "ngram_range": [1, 2],
  "model_architecture": {
    "layers": [256, 128, 64],
    "dropout_rates": [0.5, 0.4, 0.3],
    "l2_regularization": 0.001
  }
}
```

### Runtime Configuration (API)

```bash
# Set custom threshold
curl -X POST http://localhost:5000/config \
  -H "Content-Type: application/json" \
  -d '{"confidence_threshold": 0.6}'

# Get current config
curl http://localhost:5000/config
```

### Environment Variables

Create `.env` file:
```env
FLASK_ENV=production
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
MODEL_PATH=../models/expense_mlp_best.keras
CONFIDENCE_THRESHOLD=0.5
```

## 🚢 Deployment

### Local Production Server

Using **Waitress** (Windows recommended):
```bash
pip install waitress

# Run
waitress-serve --host=0.0.0.0 --port=5000 serve_model:app
```

Using **Gunicorn** (Linux/macOS):
```bash
pip install gunicorn

# Run
gunicorn --bind 0.0.0.0:5000 --workers 4 serve_model:app
```

### Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

EXPOSE 5000

CMD ["python", "src/serve_model.py"]
```

**Build and Run:**
```bash
docker build -t expense-predictor .
docker run -p 5000:5000 expense-predictor
```

### Cloud Deployment

#### Heroku
```bash
# Create Procfile
echo "web: cd src && python serve_model.py" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

#### AWS EC2 / DigitalOcean
```bash
# On server
git clone <your-repo>
cd billbuddy-deepLearning
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# Run with systemd or supervisor
waitress-serve --host=0.0.0.0 --port=5000 src.serve_model:app
```

## 🛠️ Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'flask'**
```bash
pip install flask flask-cors
```

**2. NLTK Data Not Found**
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

**3. Model File Not Found**
```bash
# Ensure you're in the src/ directory
cd src
python serve_model.py
```

**4. Low Accuracy After Training**
- Ensure `expenses_augmented.csv` has 200+ samples
- Check TF-IDF features: should be 100-200
- Verify IMPORTANT_WORDS are preserved in preprocessing
- Try increasing epochs or adjusting learning rate

**5. NumPy Version Conflicts**
```bash
pip install --upgrade numpy==1.24.3 tensorflow==2.13.0
```

### Getting Help

- 📖 See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for detailed API info
- ⚡ See [QUICK_START.md](QUICK_START.md) for setup guide
- 📝 See [SUMMARY.md](SUMMARY.md) for improvement details
- 🐛 Open an issue on GitHub

## 🤝 Contributing

Contributions are welcome! Here's how:

### 1. Fork & Clone
```bash
git clone https://github.com/YOUR_USERNAME/Expense-Category-Prediction.git
cd Expense-Category-Prediction
```

### 2. Create Branch
```bash
git checkout -b feature/your-feature-name
```

### 3. Make Changes
- Add your improvements
- Test thoroughly
- Update documentation

### 4. Test Your Changes
```bash
# Run existing tests
cd src
python test_api.py

# Test predictions
python test_predictions.py

# Ensure model trains
python train_with_augmented_data.py
```

### 5. Submit Pull Request
```bash
git add .
git commit -m "Add: your feature description"
git push origin feature/your-feature-name
```

### Contribution Ideas
- 🌍 Add support for multiple languages
- 📱 Create mobile SDK (iOS/Android)
- 🎨 Build web interface
- 📊 Add more expense categories
- 🔍 Improve preprocessing for specific domains
- 🧪 Add more test cases
- 📈 Experiment with different model architectures (BERT, transformers)

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024 BillBuddy Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🙏 Acknowledgments

- **TensorFlow** - Deep learning framework
- **scikit-learn** - Machine learning utilities
- **NLTK** - Natural language processing
- **Flask** - Web framework for API

## 📧 Contact

For questions or feedback:
- GitHub: [@ishaant97](https://github.com/ishaant97)
- Project: [Expense Category Prediction API](https://github.com/ishaant97/Expense-Category-Prediction)

---

Made with ❤️ for BillBuddy | 93.33% Accuracy 🎯

## 🐛 Troubleshooting

**Low accuracy on basic categories:**
- Run `python augment_data.py` to expand dataset
- Check `data/expenses_augmented.csv` has 150+ samples
- Ensure NLTK data is downloaded
- Verify important category words aren't removed in preprocessing

**Model always predicts same category:**
- Class imbalance - check category distribution
- Solution: The script auto-balances with class weights

**Many "Miscellaneous" predictions:**
- Lower confidence threshold (0.3-0.4)
- Add more training data
- Check if input text is too vague

**Poor performance on new/unseen categories:**
- Add examples to `data/expenses.csv`
- Update augmentation patterns
- Retrain model

## 📊 Model Artifacts

After training, these files are created in `models/`:

- `expense_mlp_best.keras` - Trained model
- `label_encoder_best.joblib` - Category encoder
- `vectorizer_best.joblib` - TF-IDF vectorizer
- `model_config.json` - Model configuration
- `training_history_best.json` - Training metrics

## 🎓 Key Learnings

1. **Dataset size matters** - 61 samples is too small for deep learning
2. **Data augmentation is crucial** for small datasets
3. **Feature engineering** should match dataset size
4. **Preprocessing** should preserve domain-specific keywords
5. **Confidence thresholds** prevent poor predictions
6. **Class balancing** improves minority class performance

## 📚 Next Steps

1. ✅ Collect more real expense data
2. ✅ Add category-specific keywords to preprocessing
3. ✅ Implement A/B testing with users
4. ✅ Monitor prediction confidence in production
5. ✅ Retrain monthly with new data
6. ✅ Add feedback loop for incorrect predictions

## 🤝 Contributing

To improve the model:
1. Add real expense examples to `data/expenses.csv`
2. Update augmentation patterns in `src/augment_data.py`
3. Tune hyperparameters in training script
4. Test with `python test_predictions.py`

---

**Questions?** Check the code comments or create an issue.

**Need help?** The training scripts have detailed logging to help debug issues.
