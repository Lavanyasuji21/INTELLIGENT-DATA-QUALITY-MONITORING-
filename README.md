# 🛡️ Intelligent Data Quality Monitoring Platform

An enterprise-grade, interactive web platform for automated data quality assessment, machine learning imputation, human-in-the-loop review, explainable deep learning anomaly detection, population drift monitoring, and comprehensive multi-sheet Excel auditing.

Built with **Streamlit**, **scikit-learn**, and **TensorFlow/Keras**, fully deployable to **Streamlit Community Cloud**.

---

## 🌟 Key Features & Pipeline

1. **Automated Data Quality Profiling & Scoring**
   - Ingests CSV, XLS, and XLSX datasets.
   - Detects missing cells, duplicated rows, empty columns, constant features, IQR outliers, negative numbers, numeric fields stored as text, and unstripped whitespace.
   - Computes an objective overall **Quality Score (%)** based on cell and row defect densities.

2. **Machine Learning Imputation (Random Forest)**
   - Utilizes `RandomForestRegressor` for continuous numerical features.
   - Utilizes `RandomForestClassifier` with balanced subsampling for categorical features.
   - Graceful fallback to median/mean/mode when observed data points are sparse.

3. **Human-in-the-Loop Governance & Manual Overrides**
   - Proposed imputation values are **never applied automatically**.
   - Review proposed vs. original datasets with side-by-side previews.
   - One-click buttons to **Accept Suggested Values** or **Reject Suggested Values**.
   - Deduplication controls: **Remove Duplicates** or **Keep Duplicates**.
   - Precise single-cell manual override by specifying row index, column name, and replacement value.

4. **Explainable Dual-Model Anomaly Detection**
   - **TensorFlow Autoencoder:** Multi-layer neural network trained on normalized numeric features to identify high reconstruction loss rows.
   - **User-Adjustable Percentile Threshold:** Default at the 95.0th percentile (adjustable from 80.0% to 99.9%).
   - **Isolation Forest:** Tree-based anomaly scoring with dynamic contamination.
   - **Explainability:** Identifies top contributing features that generated the highest reconstruction error for each flagged record.
   - Reconstruction-error distribution histogram with visual threshold markers.

5. **Multi-Metric Data Drift Monitoring**
   - Compares Reference (baseline) vs. Current (production) distributions.
   - **Population Stability Index (PSI):** Quantile-binned distribution shift measurement (Low: $<0.10$, Moderate: $0.10 \le \text{PSI} < 0.25$, High: $\ge 0.25$).
   - **Kolmogorov-Smirnov (KS) Test:** Non-parametric two-sample test for numerical features ($p < 0.05$ significance).
   - **Chi-Square Contingency Test:** Statistical test of independence for categorical distributions.
   - Horizontal color-coded severity visualization and downloadable drift reports.

6. **Before-and-After Quality Comparison**
   - Quantifies exact metric improvements between raw ingested data and the approved dataset.
   - Dual-panel chart displaying issue count reductions and quality score gains.
   - Granular metric audit table with business interpretations.

7. **Consolidated Excel Workbooks**
   - Exports multi-sheet `.xlsx` workbooks containing raw data, proposed imputations, approved datasets, detected anomalies, drift statistics, and comparison audit trails.

---

## 🏗️ Architecture & Session Isolation

The platform enforces strict multi-user session isolation:
- Each user session receives a unique session UUID and private report directory.
- `st.session_state` encapsulates instance state (`DataQualityEngine`), ensuring concurrent users cannot overwrite or access each other's datasets, decisions, or exports.
- All original algorithms, loss calculations, imputation logic, and statistical tests from the original Google Colab implementation are preserved verbatim in `backend/core_engine.py` and `original_code.py`.

---

## 📁 Repository Structure

```text
├── app.py                     # Main Streamlit web application
├── original_code.py           # Unmodified original Google Colab script
├── requirements.txt           # Python dependency specifications
├── README.md                  # Complete documentation and setup guide
├── .gitignore                 # Excludes temp files, caches, and session exports
├── .streamlit/
│   └── config.toml            # Custom light theme and server configuration
├── backend/
│   ├── core_engine.py         # Session-isolated algorithmic engine
│   └── sample_data.py         # Realistic demo data generators (quality & drift)
└── tests/
    ├── test_engine.py         # Unit tests for core algorithms
    ├── test_e2e_features.py   # Comprehensive end-to-end verification tests
    └── test_streamlit_pages.py# Automated multi-page UI rendering tests
```

---

## 📥 Quick Start: Running from Downloaded GitHub ZIP

If you or a colleague downloaded this project as a ZIP file from GitHub:

### 1. Extract the ZIP File
Extract the downloaded `intelligent-data-quality-platform-main.zip` to a folder of your choice on your computer.

### 2. Open Terminal / PowerShell in the Project Folder
Navigate into the extracted folder:
```bash
cd intelligent-data-quality-platform-main
```

### 3. Create and Activate a Python Virtual Environment
- **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Streamlit will launch automatically in your web browser at `http://localhost:8501`.

### 6. Test with Sample Data
- **Option A (Instant Demo):** Click the **⚡ Quick Load Demo Dataset** button on the Overview or Data Quality pages to load an end-to-end benchmark dataset with real defect patterns.
- **Option B (File Upload):** Upload files from the included `sample_data/` folder:
  - `sample_data/sample_quality_data.csv` (for Data Quality & Anomaly Detection)
  - `sample_data/sample_reference_drift.csv` & `sample_data/sample_current_drift.csv` (for Drift Analysis)

---

## 💻 Local Setup & Execution Instructions (Git Clone)

### 1. Prerequisites
- Python 3.10 to 3.13
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/randy792005-arch/intelligent-data-quality-platform.git
cd intelligent-data-quality-platform
```

### 3. Create and Activate a Virtual Environment
- **Windows (Command Prompt / PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the Application Locally
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

### 6. Run Automated Test Suites
```bash
python -m unittest discover tests
```

---

## 🚀 Step-by-Step Streamlit Community Cloud Deployment

### Step 1: Create a GitHub Repository
1. Navigate to [GitHub](https://github.com) and log in.
2. Click the **+** icon in the top right corner and select **New repository**.
3. Name your repository (e.g., `intelligent-data-quality-platform`).
4. Set visibility to **Public** (required for free Streamlit Community Cloud deployment).
5. Leave "Initialize this repository with a README" **unchecked** (we will push existing files).
6. Click **Create repository**.

### Step 2: Initialize Git and Push Files
In your local project folder (`data-quality-platform`), run:
```bash
git init
git add .
git commit -m "Initial commit: Intelligent Data Quality Monitoring Platform with Streamlit"
git branch -M main
git remote add origin https://github.com/<your-username>/intelligent-data-quality-platform.git
git push -u origin main
```

**Files uploaded to GitHub:**
- `app.py` *(entry point)*
- `original_code.py` *(preserved original source)*
- `requirements.txt`
- `.streamlit/config.toml`
- `backend/core_engine.py`
- `backend/sample_data.py`
- `README.md`
- `.gitignore`
- `tests/` *(optional, recommended)*

*(Note: `.cache/`, `reports/`, `uploads/`, and temporary datasets are safely excluded by `.gitignore`)*

### Step 3: Deploy on Streamlit Community Cloud
1. Go to [https://share.streamlit.io/](https://share.streamlit.io/) and sign in with your GitHub account.
2. Click **Create app** (or **New app**).
3. Under **Repository**, select `<your-username>/intelligent-data-quality-platform`.
4. Under **Branch**, select `main`.
5. Under **Main file path**, select or type `app.py`.
6. *(Optional)* Click **Advanced settings** to select Python version (Python 3.11 or 3.12 recommended).
7. Click **Deploy!**.

### Step 4: Access and Share Public URL
- Once deployment finishes (typically 2-3 minutes for TensorFlow wheel installation), your application will be live at:
  `https://<your-app-subdomain>.streamlit.app`
- Copy this public URL from your browser address bar to share in presentations, resumes, or project demonstrations.

---

## ⚙️ Cloud Resource Limits & Troubleshooting

### 1. Streamlit Community Cloud Memory Constraints
- **Resource Limits:** Streamlit Community Cloud free tier instances provide **1 GB of RAM**.
- **TensorFlow Considerations:**
  - Autoencoder model compilation and training has been pre-configured with compact architectures (`hidden_units <= 16`, `latent_units <= 8`, `batch_size <= 32`, and early stopping).
  - For standard tabular datasets (up to 50,000 cells), Autoencoder training completes within 5-15 seconds and uses under 400 MB of RAM.
  - If deploying on free cloud tiers, avoid uploading massive CSV files (> 50 MB) to prevent out-of-memory (OOM) pod evictions.

### 2. Dependency Versions
- `tensorflow>=2.15.0`: Streamlit Cloud's Ubuntu environment automatically resolves standard CPU wheels.
- `openpyxl>=3.1.0` & `xlrd>=2.0.1`: Support modern `.xlsx` and legacy `.xls` spreadsheet parsing.
- `matplotlib.use("Agg")`: Configured in both backend and app entry points to prevent GUI/display backend crashes in headless Linux server environments.

### 3. Clearing Cache & Re-deploying
- If a dependency fails during build:
  1. Open the Streamlit Cloud app dashboard.
  2. Click the three dots (⋮) next to your app and select **Reboot app** or **Clear cache and deploy**.
