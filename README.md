# Healthcare Cost Pattern Discovery & Unusual Billing Detection

An unsupervised clustering project that groups insurance policyholders by lifestyle/cost patterns using **K-Means** and detects atypical billing cases using **DBSCAN**.

---

## 📌 Project Overview
- **Objective:** Discover groups of policyholders with similar lifestyle and medical charges (K-Means), and detect unusually high or atypical cases (DBSCAN).
- **Dataset:** Medical Cost Personal Dataset (`data/insurance.csv` — 1,338 records, 7 columns).
- **Features Used:** `age`, `bmi`, `smoker` (encoded 0/1), `charges`.
- **Scaling:** `StandardScaler` so that `charges` ($1,100 to $63,000+) does not dominate Euclidean distance over `age` (18–64) and `bmi` (15–53).
- **Visualizations:** Direct physical features (**Charges vs Age** and **Charges vs BMI**). No PCA.

---

## 📂 Project Structure

```text
dwml_mini_proj/
├── data/
│   └── insurance.csv         # Raw dataset (1,338 rows)
├── models/
│   └── model.joblib          # Single serialized file with scaler, models & profiles
├── backend/
│   └── main.py               # FastAPI backend service
├── frontend/
│   └── app.py                # Streamlit web interface
├── report/
│   ├── report.html           # Printable A4 PDF report with charts & tables
│   ├── REPORT.md             # Markdown project report
│   └── report_assets/        # High-resolution chart images
├── notebook.ipynb            # Fully executed Jupyter Notebook
├── train_and_evaluate.py     # Standalone Python script to train and export model
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## ⚙️ Algorithms Explained

### 1. K-Means Clustering (Centroid-Based)
- Partitions the data into $K$ spherical clusters by minimizing within-cluster sum of squares (WCSS / Inertia).
- We tested $K=2$ to $8$ using the **Elbow Method** and **Silhouette Score**.
- **$K=4$ was selected** because it achieved the lowest **Davies–Bouldin Index ($0.9338$)** and clearly separates:
  1. **Mature Non-Smokers:** Avg Age ~53.5 yrs, Avg Charges ~$12,643 (Moderate cost).
  2. **Smokers:** Avg Age ~38.6 yrs, Avg Charges ~$32,120 (High cost cohort).
  3. **Young Lean Non-Smokers:** Avg Age ~28.8 yrs, BMI ~26.1, Avg Charges ~$5,305 (Baseline low cost).
  4. **Young High-BMI Non-Smokers:** Avg Age ~31.4 yrs, BMI ~37.2, Avg Charges ~$6,063 (Low-moderate cost).

### 2. DBSCAN (Density-Based Anomaly Detection)
- Groups points based on spatial density ($\epsilon$-neighborhood) and flags points in sparse regions as **Noise (-1)**.
- **Parameters:**
  - `min_samples = 8` ($2 \times D$ rule of thumb for 4 features).
  - `eps = 0.58`: Selected using the knee point on an **8-distance graph**.
- **Results:**
  - 5 dense clusters.
  - **63 atypical records ($4.71\%$)** flagged as noise (within the 2%–6% target).
  - Examples of flagged cases: non-smokers with sudden massive charges (>$25,000), extreme BMI (<18 or >45), or very young individuals with unexplained high bills.
- **How DBSCAN predicts new records:**
  Computes the Euclidean distance to the nearest stored DBSCAN core point. If distance $> \epsilon$ ($0.58$), the new record is marked as **Atypical Case (Outlier)**.

---

## 📊 3 Simple Evaluation Metrics

Instead of overly complex math, we use the 3 standard metrics:

| Metric | Target | K-Means ($K=4$) | DBSCAN ($\epsilon=0.58$) | What it Means |
| :--- | :---: | :---: | :---: | :--- |
| **1. Silhouette Score** | Higher ($\to +1$) | **0.3477** | **0.0614** | Measures how well-separated and cohesive clusters are (-1 to +1). |
| **2. Davies–Bouldin Index** | Lower ($\to 0$) | **0.9338** | **1.1254** | Measures overlap between clusters (0 is perfect separation). |
| **3. Noise / Outlier %** | Target: 2–6% | **0.0%** | **4.71%** | Percentage of unusual cases isolated (63 points in DBSCAN). |

---

## 🚀 Steps to Run the Project

### 1. Install Requirements
```powershell
pip install -r requirements.txt
```

### 2. Train and Save Model (Creates `models/model.joblib`)
```powershell
python train_and_evaluate.py
```

### 3. Run the Jupyter Notebook
Open `notebook.ipynb` in VS Code or Jupyter:
```powershell
jupyter notebook notebook.ipynb
```
*(All cells are pre-executed with plots of distributions, elbow curve, k-distance graph, and scatter plots).*

### 4. Run the FastAPI Backend
```powershell
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Docs: `http://127.0.0.1:8000/docs`
- Endpoints:
  - `POST /predict`: Input `age`, `bmi`, `smoker`, `charges` $\to$ Returns assigned cost group and DBSCAN outlier status.
  - `GET /summary`: Returns cluster profiles and evaluation metrics.

### 5. Run the Streamlit Frontend Web App
In another terminal:
```powershell
streamlit run frontend/app.py
```
- Interactive policyholder input form with real-time cluster persona and anomaly card.
- Displays the 3 simple metrics and interactive Plotly scatter plots (**Charges vs Age**, **Charges vs BMI**, **DBSCAN Outlier Map**).
- Works with FastAPI backend or directly with `models/model.joblib`.
