# 🛰️ AI Himawari Forecasting (Bandung Region)

Prediksi deret waktu (*time series forecasting*) suhu kecerahan awan (*Target Brightness Temperature* / TBB Band 13 - Clean IR) berbasis citra satelit **Himawari-9 (JMA)** untuk wilayah **Kota Bandung dan sekitarnya**.

Proyek ini mengintegrasikan seluruh tahapan secara otomatis (*end-to-end pipeline*): mulai dari ekstraksi data satelit NetCDF mentah, pra-pemrosesan & normalisasi, optimasi *sliding window*, *cross-validation*, pelatihan model ML/DL, peramalan multi-langkah (*iterative multi-step rollout with bias correction*), klasifikasi potensi hujan & risiko bencana hidrometeorologi, hingga visualisasi geospasial modern 6-panel serta animasi GIF.

---

## 📌 Alur Pipeline Sistem

```
Raw NetCDF (.nc)
      │
      ▼
1. loader.py           ──> Buka & validasi data NetCDF secara multiproses
      │
      ▼
2. preprocessing.py    ──> Crop ROI Bandung, scaling fitur (Standard/MinMax)
      │
      ▼
3. experiment.py       ──> Pencarian window size terbaik (Window Size Search)
      │
      ▼
4. windowing.py        ──> Pembentukan sliding window (X_window, y_window)
      │
      ▼
5. dataset_builder.py  ──> Subsampling & split train/test bebas kebocoran (leak-free)
      │
      ▼
6. experiment.py       ──> Time-Series Blocked K-Fold Cross Validation (CV)
      │
      ▼
7. training.py         ──> Fitting model terpilih & persistensi (.joblib)
      │
      ▼
8. forecasting.py      ──> Peramalan multi-langkah (Bias Correction / Teacher Forcing)
      │
      ▼
9. classification.py   ──> Klasifikasi Awan (Hujan/Mendung/Cerah) & Risiko (Bahaya/Waspada/Aman)
      │
      ▼
10. evaluation.py      ──> Evaluasi metrik regresi (RMSE, MAE, MAPE, R²) & klasifikasi (F1, Conf. Matrix)
      │
      ▼
11. export.py          ──> Simpan artifak komprehensif (.json, .csv, .npy)
      │
      ▼
12. visualization.py   ──> Render Dashboard 6-Panel & Animasi GIF dengan overlay batas Kota Bandung (.geojson)
```

---

## 📁 Struktur Direktori Proyek

```text
AI_HIMAWARI_updated/
│
├── config.py                 # Konfigurasi terpusat (parameter, path, toggle pipeline, threshold)
├── main.py                   # Orchestrator utama pemanggilan seluruh pipeline
│
├── loader.py                 # Loader file NetCDF Himawari-9 paralel multiprocessing
├── preprocessing.py          # Cropping spasial ROI Bandung & scaling data
├── windowing.py              # Pembuatan time-series sliding window & flattening
├── dataset_builder.py        # Pembagian train/test & spatial/temporal subsampling
├── experiment.py             # Optimasi window size search & K-Fold cross validation
├── training.py               # Pelatihan model & penyimpanan artifact model (.joblib)
├── forecasting.py            # Mesin inferensi time-series & koreksi bias
├── classification.py         # Pengkategorian TBB ke kondisi awan & tingkat bahaya
├── evaluation.py             # Kalkulasi metrik performa regresi & klasifikasi
├── export.py                 # Ekspor hasil ke format CSV, JSON, NumPy
├── visualization.py          # Visualisasi heatmap geospasial Bandung, Dashboard 6-Panel, & GIF
├── run_metrics.py            # Tracking durasi proses, penggunaan memori (GB), & CPU
│
├── models/                   # Layer Model Abstrak & Implementasi Konkret
│   ├── base.py               # Interface kontrak dasar (BaseModel)
│   ├── registry.py           # Registri pemetaan nama model -> class
│   ├── ann.py                # Implementasi model Artificial Neural Network (MLPRegressor)
│   ├── gradient_boosting.py  # Implementasi Gradient Boosting Regressor
│   └── gaussian_process.py   # Implementasi Gaussian Process Regressor
│
├── KotaBandung.geojson       # Batas administratif Kota Bandung untuk masking & outline visualisasi
├── requirements.txt          # Daftar dependensi Python
├── .gitignore                # File filter git (mencegah data/output besar masuk repositori)
└── README.md                 # Dokumentasi proyek
```

---

## 🚀 Instalasi & Persiapan Lingkungan

### 1. Prasyarat
- Python 3.9+ (direkomendasikan Python 3.10 atau 3.11)
- Git

### 2. Clone Repositori & Buat Virtual Environment

```bash
# Clone repository
git clone https://github.com/USERNAME/AI_HIMAWARI.git
cd AI_HIMAWARI

# Buat virtual environment
python -m venv venv

# Aktifkan virtual environment
# Pada Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Pada Windows (CMD):
.\venv\Scripts\activate.bat
# Pada Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependensi

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Struktur Data Mentah NetCDF
Letakkan dataset NetCDF Himawari-9 di dalam folder `data/` sesuai format struktur folder tanggal BMKG/JMA:
```text
data/
└── data_bandung/
    └── jma/
        └── netcdf/
            ├── 202512/
            │   ├── 01/
            │   │   └── NC_H09_20251201_0000_R21_FLDK.02401_02401.nc
            │   └── ...
            └── 202601/
                └── ...
```
> *(Catatan: Folder `data/` dan `outputs/` sudah otomatis dikecualikan oleh `.gitignore` sehingga tidak akan terunggah ke GitHub).*

---

## ⚙️ Konfigurasi (`config.py`)

Seluruh pengaturan dapat disesuaikan pada [config.py](file:///d:/Himawari_real/AI_HIMAWARI_updated/config.py):

- **Model Utama**: Atur `DEFAULT_MODEL = "ann"` (pilihan: `"ann"`, `"gradient_boosting"`, `"gaussian_process"` atau model baru Anda).
- **Periode Bulan**: 
  - `TRAIN_MONTHS = ("202512", "202601", "202602", ...)`
  - `TEST_MONTHS = ("202606", "202607")`
- **Toggle Tahapan Pipeline**:
  - `RUN_EXPERIMENT = True` (mencari window size optimal 1–24 langkah)
  - `RUN_CROSS_VALIDATION = True` (menjalankan 5-Fold Time-Series CV)
  - `RUN_TRAINING = True` (melatih model)
  - `RUN_FORECAST = True` (menghasilkan prediksi test)
  - `RUN_CLASSIFICATION = True` (klasifikasi hujan & risiko)
  - `SAVE_GIF = True` (menghasilkan visualisasi animasi GIF)

---

## ▶️ Menjalankan Proyek

Cukup jalankan satu perintah:

```bash
python main.py
```

Seluruh artifak hasil eksekusi (model tersimpan, file evaluasi JSON/CSV, gambar dashboard perbandingan, dan animasi GIF) akan tersimpan di dalam:
```text
outputs/<EXPERIMENT_NAME>_<EXPERIMENT_VERSION>/
├── summary/              # summary.json & log eksekusi
├── models/               # model.joblib
├── forecast/             # forecast_test.npy
├── classification/       # cloud_class.npy & risk_class.npy
├── evaluation/           # regression_metrics.csv, classification_metrics.csv
└── visualization/        # dashboard_six_panel.png, forecast_rollout.gif, cm_*.png
```

---

## 🛠️ Panduan Menambahkan Model Baru di `models/`

Proyek ini menggunakan arsitektur **Model Agnostic** berbasis Factory Pattern (`models/registry.py`). Modul pelatihan (`training.py`), eksperimen (`experiment.py`), dan peramalan (`forecasting.py`) tidak pernah terikat langsung pada kelas model tertentu.

Untuk menambahkan model baru (misalnya **Random Forest**, **XGBoost**, **LightGBM**, atau **LSTM/GRU**), ikuti 3 langkah mudah berikut:

### Langkah 1: Buat File Model di `models/<nama_model>.py`
Wajib mewarisi (*inherit*) kelas abstrak `BaseModel` dari [models/base.py](file:///d:/Himawari_real/AI_HIMAWARI_updated/models/base.py) dan mengimplementasikan atribut serta metode:
- `requires_flat_input: bool`:
  - `True`: jika model membutuhkan input 2D `(n_samples, window_size * n_features)` seperti algoritma tabular/ML standar (Random Forest, SVR, MLP, XGBoost).
  - `False`: jika model membutuhkan input 3D sekuensial `(n_samples, window_size, n_features)` seperti RNN, LSTM, GRU, atau Transformer.
- `fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None`
- `predict(self, X: np.ndarray) -> np.ndarray`

**Contoh Template Kode (`models/random_forest.py`):**
```python
"""Random Forest model implementation for AI Himawari Forecasting."""
from __future__ import annotations
import logging
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from config import RANDOM_SEED
from models.base import BaseModel

logger = logging.getLogger(__name__)

class RandomForestModel(BaseModel):
    """Random Forest forecasting model."""

    # Gunakan True untuk input 2D (tabular)
    requires_flat_input: bool = True

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = 10,
        random_state: int = RANDOM_SEED,
        **kwargs: object,
    ) -> None:
        self._model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            **kwargs,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._validate_2d(X_train)
        logger.info("Fitting RandomForestModel pada %d sampel, dimensi %s.", X_train.shape[0], X_train.shape)
        self._model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._validate_2d(X)
        return self._model.predict(X)
```

---

### Langkah 2: Daftarkan di `models/registry.py`
Buka [models/registry.py](file:///d:/Himawari_real/AI_HIMAWARI_updated/models/registry.py):
1. Import kelas model baru Anda:
   ```python
   from models.random_forest import RandomForestModel
   ```
2. Tambahkan ke dictionary `_REGISTRY`:
   ```python
   _REGISTRY: dict[str, type[BaseModel]] = {
       "ann": ANNModel,
       "gradient_boosting": GradientBoostingModel,
       "gaussian_process": GaussianProcessModel,
       "random_forest": RandomForestModel,   # <-- Tambahkan ini
   }
   ```

---

### Langkah 3: Tambahkan ke `config.py`
Buka [config.py](file:///d:/Himawari_real/AI_HIMAWARI_updated/config.py):
1. Tambahkan string identifier model ke `SUPPORTED_MODELS`:
   ```python
   SUPPORTED_MODELS: tuple[str, ...] = (
       "ann",
       "gaussian_process",
       "gradient_boosting",
       "random_forest",   # <-- Tambahkan ini
   )
   ```
2. (Opsional) Jika ingin langsung menjalankan eksperimen dengan model tersebut, atur:
   ```python
   DEFAULT_MODEL: str = "random_forest"
   EXPERIMENT_NAME: str = "random_forest"
   ```

Selesai! Pipeline otomatis mengenali model baru Anda tanpa perlu mengubah satu baris pun kode di `main.py`, `training.py`, atau `forecasting.py`.

---

## 🤖 Template Prompt untuk Rekan Tim (Jika Memakai AI)

Jika rekan Anda ingin meminta bantuan AI (seperti ChatGPT, Claude, atau Gemini) untuk membuatkan kode arsitektur model baru, berikan prompt siap pakai di bawah ini:

---

### 📋 Salin Prompt Ini ke AI:

```text
Halo! Saya sedang mengembangkan proyek "AI Himawari Satellite Time Series Forecasting" menggunakan Python.
Proyek ini memiliki arsitektur modular di mana semua model peramalan wajib mengimplementasikan interface kontrak abstrak `BaseModel`.

Tolong buatkan implementasi model baru untuk arsitektur: [NAMA MODEL, misal: LightGBM / XGBoost / PyTorch LSTM / CatBoost / SVR] yang disimpan di dalam file: `models/[nama_file].py`.

Berikut adalah aturan dan kontrak yang WAJIB dipenuhi:
1. Model wajib berupa class yang mewarisi (inherit) `BaseModel` dari `models.base`.
2. Atribut `requires_flat_input`:
   - Set ke `True` jika model menerima input 2D shape (n_samples, window_size * n_features), seperti model scikit-learn/tabular.
   - Set ke `False` jika model menerima input sekuensial 3D shape (n_samples, window_size, n_features), seperti LSTM/GRU.
3. Method `fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None`:
   - Melatih model dengan input fitur X_train dan target y_train berdimensi (n_samples,).
   - Jika requires_flat_input == True, panggil `self._validate_2d(X_train)`.
4. Method `predict(self, X: np.ndarray) -> np.ndarray`:
   - Melakukan prediksi dan mengembalikan array 1D numpy dengan shape (n_samples,).
   - Jika requires_flat_input == True, panggil `self._validate_2d(X)`.
5. Gunakan `RANDOM_SEED` dari `config.py` sebagai default random state agar eksperimen reproducible.
6. Sertakan type hints lengkap, docstrings yang rapi, dan logging menggunakan `logging.getLogger(__name__)`.

Berikan:
1. Kode lengkap untuk `models/[nama_file].py`.
2. Baris import dan pendaftaran yang perlu saya tambahkan ke `models/registry.py`.
3. String identifier yang perlu saya tambahkan ke `SUPPORTED_MODELS` di `config.py`.
```

---

## 👥 Kontributor & Lisensi
Proyek ini dikembangkan untuk penelitian dan operasional peramalan cuaca berbasis citra satelit Himawari-9 BMKG/JMA di wilayah cekungan Bandung.
