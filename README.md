# Bank Transaction Dataset for Fraud Detection
## Project Machine Learning - Deteksi Penipuan Transaksi Bank

### 📋 Deskripsi Project

Project ini merupakan implementasi Machine Learning untuk **deteksi penipuan (fraud detection)** pada transaksi perbankan menggunakan **Bank Transaction Dataset**. Dataset ini berisi 2.512 sampel data transaksi dengan berbagai atribut seperti jumlah transaksi, tipe transaksi, lokasi, umur customer, dan lain-lain.

Project ini terdiri dari dua tahap utama:
1. **Clustering** - Mengelompokkan data transaksi untuk identifikasi pola
2. **Klasifikasi** - Membangun model untuk memprediksi transaksi yang berpotensi fraud

---

### 📁 Struktur Project

```
Machine Learning Submission/
├── [Clustering]_Submission_Akhir_BMLP_Your_Name.ipynb
├── [Klasifikasi]_Submission_Akhir_BMLP_Your_Name.ipynb
├── data_clustering.csv
├── data_clustering_inverse.csv
├── PCA_model_clustering.h5
├── model_clustering.h5
├── decision_tree_model.h5
├── explore_<Classification>_classification.h5
├── tuning_classification.h5
└── README.md
```

---

### 📊 Dataset

**Sumber Data**: Bank Transaction Dataset
- **Total Sampel**: 2.512 transaksi
- **Fitur Utama**:
  - `TransactionID`: ID unik transaksi
  - `AccountID`: ID akun
  - `TransactionAmount`: Jumlah transaksi
  - `TransactionDate`: Tanggal transaksi
  - `TransactionType`: Tipe transaksi (Credit/Debit)
  - `Location`: Lokasi geografis
  - `DeviceID`: ID perangkat
  - `IP Address`: Alamat IP
  - `MerchantID`: ID merchant
  - `AccountBalance`: Saldo akun
  - `Channel`: Kanal transaksi (Online/ATM/Branch)
  - `CustomerAge`: Umur customer
  - `CustomerOccupation`: Pekerjaan (Doctor/Engineer/Student/Retired)
  - `TransactionDuration`: Durasi transaksi (detik)
  - `LoginAttempts`: Jumlah upaya login

---

### 🔬 Tahap 1: Clustering

**File**: `[Clustering]_Submission_Akhir_BMLP_Your_Name.ipynb`

#### Proses yang Dilakukan:
1. **Import Library**
   - pandas, numpy, matplotlib, seaborn
   - sklearn (LabelEncoder, StandardScaler, KMeans, PCA)
   - yellowbrick, joblib

2. **Load & Exploratory Data Analysis**
   - Memuat dataset dari Google Sheets
   - Analisis struktur data dengan `head()`, `info()`, `describe()`
   - Visualisasi distribusi data

3. **Data Preprocessing**
   - Handling missing values
   - Feature engineering (binning CustomerAge)
   - Label Encoding untuk data kategorikal
   - Feature Scaling dengan StandardScaler

4. **Dimensionality Reduction**
   - Implementasi PCA (Principal Component Analysis)
   - Mengurangi dimensi data untuk efisiensi clustering

5. **Clustering dengan K-Means**
   - Menentukan jumlah cluster optimal dengan Elbow Method
   - Evaluasi menggunakan Silhouette Score
   - Interpretasi hasil clustering

6. **Output**
   - `data_clustering.csv` - Data dengan hasil clustering
   - `data_clustering_inverse.csv` - Data dengan inverse transform (Advanced)
   - `model_clustering.h5` - Model clustering tersimpan
   - `PCA_model_clustering.h5` - Model PCA tersimpan

---

### 🎯 Tahap 2: Klasifikasi

**File**: `[Klasifikasi]_Submission_Akhir_BMLP_Your_Name.ipynb`

#### Proses yang Dilakukan:
1. **Import Library**
   - pandas
   - sklearn (train_test_split, DecisionTreeClassifier, RandomForestClassifier)
   - sklearn.metrics (accuracy, precision, recall, f1_score)
   - GridSearchCV, RandomizedSearchCV

2. **Load Data**
   - Memuat hasil clustering (`data_clustering_inverse.csv`)
   - Data sudah memiliki fitur `Target` untuk supervised learning

3. **Feature Encoding (Opsional)**
   - One Hot Encoding untuk fitur kategorikal
   - Dilakukan jika menggunakan data inverse transform

4. **Data Splitting**
   - Membagi data menjadi training (80%) dan testing (20%)
   - Stratified split untuk menjaga proporsi kelas

5. **Model Building**
   - **Base Model**: Decision Tree Classifier
   - **Model Tambahan**: Random Forest Classifier (opsional)

6. **Model Evaluation**
   - Accuracy Score
   - Precision Score
   - Recall Score
   - F1 Score
   - Classification Report

7. **Hyperparameter Tuning**
   - GridSearchCV atau RandomizedSearchCV
   - Optimasi parameter model

8. **Model Saving**
   - `decision_tree_model.h5` - Model Decision Tree
   - `explore_<Classification>_classification.h5` - Model eksplorasi
   - `tuning_classification.h5` - Model hasil tuning

---

### 🚀 Cara Menjalankan Project

#### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn yellowbrick joblib
```

#### Menjalankan Clustering
1. Buka notebook `[Clustering]_Submission_Akhir_BMLP_Your_Name.ipynb`
2. Run all cells secara berurutan
3. File output akan tersimpan otomatis

#### Menjalankan Klasifikasi
1. Pastikan tahap clustering sudah selesai
2. Buka notebook `[Klasifikasi]_Submission_Akhir_BMLP_Your_Name.ipynb`
3. Run all cells secara berurutan
4. Model akan tersimpan dalam format .h5

---

### 📈 Hasil dan Evaluasi

#### Clustering
- **Metode**: K-Means Clustering
- **Evaluasi**: Silhouette Score
- **Jumlah Cluster**: [Disesuaikan berdasarkan Elbow Method]

#### Klasifikasi
Model dievaluasi menggunakan metrik:
- **Accuracy**: Persentase prediksi yang benar
- **Precision**: Kemampuan model menghindari false positive
- **Recall**: Kemampuan model mendeteksi semua kasus positif
- **F1-Score**: Harmonic mean dari precision dan recall

---

### 🔍 Insight dan Interpretasi

**Clustering Analysis:**
- Clustering membantu mengidentifikasi pola transaksi yang berbeda
- Setiap cluster merepresentasikan kelompok transaksi dengan karakteristik serupa
- Hasil clustering dijadikan fitur tambahan untuk model klasifikasi

**Classification Analysis:**
- Model dapat memprediksi transaksi yang berpotensi fraud
- Feature importance menunjukkan variabel yang paling berpengaruh
- Hyperparameter tuning meningkatkan performa model

---

### 📝 Catatan Penting

1. **Dataset**: Dataset dimuat langsung dari Google Sheets
2. **Missing Values**: Sudah ditangani pada tahap preprocessing
3. **Feature Engineering**: Dilakukan binning pada fitur CustomerAge
4. **Model Serialization**: Model disimpan dalam format .h5 menggunakan joblib
5. **Reproducibility**: Menggunakan `random_state=42` untuk reproduksi hasil

---

### 👨‍💻 Author

**Your Name**
- Project: Bank Transaction Fraud Detection
- Course: Belajar Machine Learning untuk Pemula (BMLP)

---

### 📚 Referensi

- Scikit-learn Documentation: https://scikit-learn.org/
- K-Means Clustering: https://scikit-learn.org/stable/modules/clustering.html
- Decision Tree Classification: https://scikit-learn.org/stable/modules/tree.html
- Dataset Source: [Bank Transaction Dataset]

---

### 📄 License

This project is created for educational purposes as part of Machine Learning course submission.

---

**Last Updated**: 2026-01-12
