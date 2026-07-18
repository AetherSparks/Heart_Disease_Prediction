# Heart Disease Prediction

A Flask web application that predicts the presence of heart disease using 8 machine learning models. Users input 13 clinical features via individual form fields and get predictions from any chosen model, along with its performance metrics.

## Tech Stack

- **Backend:** Python 3.9+, Flask, Gunicorn
- **ML Libraries:** scikit-learn, XGBoost
- **Frontend:** HTML, Bootstrap 4 / Tailwind CSS
- **Dataset:** [UCI Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) (1025 records → 302 unique after dedup)

## Dataset

The dataset is automatically downloaded from Kaggle via `kagglehub` when you run `train.py`.

- **Raw:** 1025 rows (723 are exact duplicates from aggregated UCI sources)
- **After dedup:** 302 unique patient records
- **After augmentation:** ~1000 rows (original + synthetic variations)
- **Features:** 13 clinical attributes + binary target

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `age` | Continuous | Age in years |
| 2 | `sex` | Binary | 1 = male, 0 = female |
| 3 | `cp` | Ordinal | Chest pain type (1-4) |
| 4 | `trestbps` | Continuous | Resting blood pressure (mm Hg) |
| 5 | `chol` | Continuous | Serum cholesterol (mg/dl) |
| 6 | `fbs` | Binary | Fasting blood sugar > 120 mg/dl |
| 7 | `restecg` | Ordinal | Resting ECG results (0-2) |
| 8 | `thalach` | Continuous | Max heart rate achieved |
| 9 | `exang` | Binary | Exercise induced angina |
| 10 | `oldpeak` | Continuous | ST depression induced by exercise |
| 11 | `slope` | Ordinal | ST segment slope (0-2) |
| 12 | `ca` | Ordinal | Major vessels colored (0-4) |
| 13 | `thal` | Nominal | Thalassemia (3, 6, 7) |
| **Target** | `target` | Binary | 0 = no disease, 1 = disease present |

## Models & Performance

All models evaluated on a held-out original test set (61 records). Training data augmented from 241 → 1000 rows.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Neural Network (MLP) | **86.89%** | 86.84% | 91.67% | **89.19%** | **93.67%** |
| Support Vector Machine | 83.61% | 84.21% | 88.89% | 86.49% | 88.67% |
| Random Forest | 83.61% | 84.21% | 88.89% | 86.49% | 90.61% |
| XGBoost | 83.61% | 86.11% | 86.11% | 86.11% | 90.22% |
| Logistic Regression | 81.97% | 82.05% | 88.89% | 85.33% | 88.33% |
| Decision Tree | 81.97% | 79.07% | 94.44% | 86.08% | 79.22% |
| Naive Bayes | 80.33% | 81.58% | 86.11% | 83.78% | 88.67% |
| K-Nearest Neighbors | 55.74% | 61.54% | 66.67% | 64.00% | 57.00% |

## Training Pipeline (`train.py`)

1. **Download** dataset from Kaggle (1025 rows)
2. **Deduplicate** → 302 unique records
3. **Split** 80/20 → 241 train + 61 test (test is original-only for fair evaluation)
4. **Augment** training set to 1000 rows using controlled noise injection:
   - Continuous features: gaussian noise at 10% of std, clipped to data bounds
   - Binary features: flipped with 10% probability
   - Ordinal features: shifted ±1 with 15% probability
   - Nominal features: swapped to another valid value with 10% probability
5. **Train** all 8 models on augmented data
6. **Evaluate** on original test set (6 metrics each)
7. **Export** results & graphs to `static/results/`

## Results & Graphs

Generated automatically by `train.py` and saved to `static/results/`:

| File | Description |
|------|-------------|
| `comparison.csv` | All metrics for all 8 models |
| `comparison_line.png` | Multi-line chart (models × metrics) |
| `comparison_bar.png` | Grouped bar chart comparison |
| `confusion_matrices.png` | 8-panel confusion matrix grid |

These are viewable at the `/models` route in the web app.

## Project Structure

```
HeartDiseasePrediction/
├── app.py                     # Flask web app (2 routes: /, /models)
├── train.py                   # Training pipeline (download, dedup, augment, train, evaluate)
├── requirements.txt           # Minimal Python dependencies
├── vercel.json                # Vercel deployment config
├── .python-version            # Python version for Vercel
├── .gitattributes             # Line ending normalization
├── .gitignore
├── README.md
├── templates/
│   ├── index.html             # Prediction form (13 individual fields)
│   └── models.html            # Model performance dashboard with graphs
├── models/                    # 8 pre-trained model files (.pkl)
├── data/                      # Dataset cache (gitignored)
├── static/results/            # Generated metrics & graphs (gitignored)
├── flaskvenv/                 # Virtual environment (gitignored)
└── .vscode/
    └── launch.json            # VS Code debugger configs
```

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip
- Kaggle API key (for `kagglehub` — kaggle.json in `~/.kaggle/`)

### 1. Setup

```bash
git clone https://github.com/AetherSparks/Heart_Disease_Prediction.git
cd Heart_Disease_Prediction
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Train Models

Downloads dataset, deduplicates, augments, trains all 8 models, and exports results:

```bash
python train.py
```

### 3. Run Web App

```bash
python app.py
```

Open http://127.0.0.1:5000. Fill the 13 individual form fields (or click **Generate Random Data**), pick a model, and predict.

Visit http://127.0.0.1:5000/models to see the full model comparison dashboard with graphs.

## Deployment

### Option 1: Vercel (Recommended)

This project is pre-configured for Vercel:

1. Push to GitHub:
   ```bash
   git push -u origin main
   ```
2. Go to [vercel.com/new](https://vercel.com/new)
3. Import your GitHub repository
4. Vercel auto-detects Flask — deploy with zero config
5. Set environment variable `PYTHON_VERSION=3.12` if needed

**Vercel limits:**
- Bundle size: 500MB (our deps are ~200MB without TensorFlow/PyTorch)
- Cold starts: 1-3s on Fluid compute
- Free tier: generous, but large ML deps may push limits

**Important:** Run `python train.py` locally and push the `models/` directory with the pre-trained `.pkl` files. Vercel doesn't run training — it serves predictions.

### Option 2: PythonAnywhere (Free, Always-On)

Better for ML apps since it's purpose-built for Python and doesn't sleep:

1. Create account at [pythonanywhere.com](https://www.pythonanywhere.com)
2. Open a Bash console and clone:
   ```bash
   git clone https://github.com/AetherSparks/Heart_Disease_Prediction.git
   ```
3. Create a virtualenv, install deps, upload the `models/` folder
4. Set up a Web app → Manual config → Python/Flask
5. Point WSGI at `app.py`

**Limits:** 100 CPU-seconds/day, 512MB storage. Fine for lightweight usage.

### Option 3: Render (Free, Spins Down After Inactivity)

1. Push to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect repo, set start command: `gunicorn app:app`
4. Free tier spins down after 15min of inactivity (30-50s cold start)

## VS Code Debugging

Three launch configurations are provided in `.vscode/launch.json`:

| Config | Purpose |
|--------|---------|
| **Run Flask App** | Debug `app.py` with Jinja template support |
| **Run Train Script** | Debug `train.py` |
| **Current File** | Debug whatever file is open |

Press `F5` to start debugging with the selected config.

## Git

```bash
git remote -v                   # Check remote
git log --oneline --graph       # View history
git push -u origin main         # Push to GitHub
```
