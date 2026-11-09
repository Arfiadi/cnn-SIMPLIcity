# SIMPLIcity CNN Image Classification

> **Mata Kuliah:** Neuro Computing / Deep Learning (PENS)  
> **Studi Kasus:** Image Classification pada SIMPLIcity Dataset (1.000 citra, 10 kategori)  
> **Workflow:** Modular coding di lokal (CPU smoke test) & akselerasi training di Google Colab / Kaggle (GPU).

---

## 📌 Gambaran Umum Proyek

Repositori ini mengimplementasikan eksperimen klasifikasi citra berbasis Convolutional Neural Network (CNN) pada **Dataset SIMPLIcity** dengan arsitektur modular, portabel (*device-agnostic* & *path-agnostic*), dan terstruktur rapi.

### Fitur Utama:
- **Portabilitas Penuh:** Otomatis mendeteksi akselerator CUDA/GPU pada Colab/Kaggle dan beralih aman ke CPU pada laptop lokal tanpa error.
- **Path-Agnostic:** Resolusi jalur dataset otomatis antara direktori kerja lokal, `/content/` di Colab, atau `/kaggle/input/` di Kaggle.
- **Model yang Didukung:**
  1. `baseline_custom`: Arsitektur 2-layer CNN baseline sesuai spesifikasi Slide 6 PDF penugasan.
  2. `custom_scratch`: Arsitektur CNN mandiri yang dilengkapi *Batch Normalization*, *Dropout*, dan *Adaptive Average Pooling* untuk menekan *overfitting* pada dataset berukuran kecil ($N=1.000$).
  3. `resnet18` & `mobilenet_v3_small`: Benchmark transfer learning (*ImageNet pretrained*).
- **Tracking & Visualisasi Lengkap:** Menghasilkan kurva konvergensi gradient descent (*Loss & Accuracy Curve*) serta matriks kontingensi (*Confusion Matrix*) 10 kelas secara otomatis.
- **Audit Integritas Data:** Skrip verifikasi format citra, deteksi file korup, dan pengecekan distribusi per kelas.

---

## 📂 Struktur Direktori

```text
├── configs/
│   ├── default_config.yaml       # Konfigurasi standar training
│   └── smoke_test.yaml           # Konfigurasi verifikasi mikro di CPU
├── Dataset/
│   └── SIMPLIcity/
│       └── SIMPLIcity Dataset/
│           ├── train/            # 500 citra (50 per kelas: x00 - x49)
│           ├── validation/       # 300 citra (30 per kelas: x50 - x79)
│           └── test/             # 200 citra (20 per kelas: x80 - x99)
├── notebooks/
│   └── colab_kaggle_runner.ipynb # Notebook siap pakai untuk Colab/Kaggle
├── scripts/
│   ├── audit_data.py             # Audit integritas dan dimensi dataset
│   ├── package_dataset.py        # Pengemasan dataset menjadi zip portabel
│   ├── train.py                  # Entrypoint utama training & evaluasi
│   └── evaluate.py               # Evaluasi checkpoint independen
├── src/
│   └── simplicity_cnn/
│       ├── data/                 # Dataset loader, transforms v2, audit logic
│       ├── models/               # Baseline, custom scratch, transfer backbones
│       ├── training/             # Trainer, mixed precision, tracker kurva
│       ├── evaluation/           # Metrik evaluasi multi-kelas & confusion matrix
│       └── utils/                # Deteksi runtime, seeding, logging
├── tests/
│   └── test_pipeline.py          # Unit test deterministik
├── pyproject.toml                # Konfigurasi build & dependensi
└── requirements.txt              # Daftar dependensi pip
```

---

## 🚀 Panduan Penggunaan Lokal

### 1. Setup Environment Virtual (Direkomendasikan via `uv`)

```bash
# Membuat virtual environment
uv venv

# Menginstal dependensi proyek dan dev tools
uv pip install -e ".[dev]"
```

Atau menggunakan `pip` standar:
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### 2. Audit Integritas Dataset

```bash
python scripts/audit_data.py
```

### 3. Eksekusi Smoke Test di CPU Lokal

Jalankan pengujian mikro (2 epoch, subset mini) untuk memastikan seluruh pipeline berfungsi tanpa membebani laptop:

```bash
python scripts/train.py --smoke-test
```

### 4. Menjalankan Unit Tests

```bash
pytest tests/ -v
```

---

## ☁️ Panduan Training di Google Colab / Kaggle (GPU)

1. **Paketkan Dataset dari Lokal:**
   ```bash
   python scripts/package_dataset.py --output-zip simplicity_dataset.zip
   ```
2. **Buka Notebook Runner:**
   - Unggah repositori ke GitHub atau upload source code ke Colab.
   - Buka `notebooks/colab_kaggle_runner.ipynb` di Google Colab.
   - Pastikan runtime diset ke **T4 GPU** (`Runtime -> Change runtime type -> T4 GPU`).
3. **Eksekusi Eksperimen Komparasi:**
   - Sel di notebook akan mengekstrak dataset dan menjalankan perintah training:
     ```bash
     # Baseline Model (Slide 6)
     python scripts/train.py --model baseline_custom --epochs 30 --batch-size 32

     # Regularized Custom Model from Scratch
     python scripts/train.py --model custom_scratch --epochs 35 --batch-size 32

     # Transfer Learning Benchmark
     python scripts/train.py --model resnet18 --epochs 20 --batch-size 32
     ```
   - Sel evaluasi akan menampilkan grafik kurva gradient descent (`gradient_descent_summary.png`) dan matriks kontingensi (`confusion_matrix.png`) langsung pada notebook.

---

## 📊 Metrik & Label Kelas

10 kategori pada SIMPLIcity:
| Index | Label | Rentang File Asli |
|---|---|---|
| 0 | People | 0 - 99 |
| 1 | Beach | 100 - 199 |
| 2 | Building | 200 - 299 |
| 3 | Bus | 300 - 399 |
| 4 | Dinosaur | 400 - 499 |
| 5 | Elephant | 500 - 599 |
| 6 | Flower | 600 - 699 |
| 7 | Horse | 700 - 799 |
| 8 | Mountain | 800 - 899 |
| 9 | Food | 900 - 999 |
