
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

# 1. Membaca dataset
df = pd.read_csv("data_mahasiswa.csv")

fitur = [
    "jam_belajar",
    "kehadiran_persen",
    "nilai_tugas"
]
target = "kategori_performa"

print("=== DATASET ===")
print(df.head())

print("\nJumlah data:", len(df))
print("Data kosong:\n", df.isnull().sum())
print("Data duplikat:", df.duplicated().sum())

# 2. Memastikan data siap digunakan
if df[fitur + [target]].isnull().any().any():
    raise ValueError("Dataset memiliki data kosong.")

if not df["kehadiran_persen"].between(0, 100).all():
    raise ValueError("Persentase kehadiran harus 0-100.")

if not df["nilai_tugas"].between(0, 100).all():
    raise ValueError("Nilai tugas harus 0-100.")

X = df[fitur]
y = df[target]

# 3. Membagi data untuk pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nJumlah data latih:", len(X_train))
print("Jumlah data uji:", len(X_test))

# 4. Membuat dan melatih model Decision Tree
model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

# 5. Mengevaluasi model
prediksi_uji = model.predict(X_test)

akurasi = accuracy_score(y_test, prediksi_uji)

print("\n=== EVALUASI MODEL ===")
print(f"Akurasi: {akurasi:.2%}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    prediksi_uji,
    zero_division=0
))

# 6. Menyimpan grafik evaluasi
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    prediksi_uji,
    ax=axes[0],
    colorbar=False
)
axes[0].set_title("Confusion Matrix")

plot_tree(
    model,
    feature_names=fitur,
    class_names=model.classes_,
    filled=True,
    rounded=True,
    ax=axes[1]
)
axes[1].set_title("Decision Tree Model")

plt.tight_layout()
plt.savefig("hasil_evaluasi.png", dpi=150)
plt.close()

# 7. Memprediksi data mahasiswa baru (simulasi)
mahasiswa_baru = pd.DataFrame([{
    "jam_belajar": 5,
    "kehadiran_persen": 85,
    "nilai_tugas": 82
}])

hasil = model.predict(mahasiswa_baru)[0]
probabilitas = model.predict_proba(mahasiswa_baru)[0].max()

print("\n=== PREDIKSI DATA BARU ===")
print(mahasiswa_baru.to_string(index=False))
print("Kategori prediksi:", hasil)
print(f"Probabilitas kelas menurut model: {probabilitas:.2%}")

# 8. Menyimpan hasil prediksi dan evaluasi
hasil_baru = mahasiswa_baru.copy()
hasil_baru["kategori_prediksi"] = hasil
hasil_baru["probabilitas_kelas"] = round(
    probabilitas, 4
)
hasil_baru.to_csv("hasil_prediksi.csv", index=False)

print("\nFile hasil_prediksi.csv berhasil dibuat.")
print("File hasil_evaluasi.png berhasil dibuat.")
