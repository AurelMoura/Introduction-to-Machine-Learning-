# Introduction-to-Machine-Learning-
# Student Academic Performance Classification

## 1. Deskripsi Proyek

Proyek ini merupakan implementasi materi Praktisi Mengajar berjudul *Introduction to Machine Learning* yang disampaikan oleh Taniza Tajudin.

Proyek bertujuan menerapkan algoritma Machine Learning untuk mengklasifikasikan kategori performa akademik berdasarkan data simulasi jam belajar, persentase kehadiran, dan nilai tugas.

## 2. Tujuan

- Memahami tahapan dasar Machine Learning.
- Mempersiapkan dan memeriksa kualitas dataset.
- Membagi data menjadi data latih dan data uji.
- Melatih model menggunakan algoritma Decision Tree.
- Mengevaluasi kinerja model klasifikasi.
- Mencoba prediksi pada data baru.

## 3. Tools yang Digunakan

- Python 3.13
- Pandas
- Scikit-learn
- Matplotlib
- Visual Studio Code

## 4. Dataset

Dataset berisi 36 baris data simulasi dengan fitur jam belajar, persentase kehadiran, nilai tugas, dan target kategori performa.

Kategori terdiri dari Perlu Peningkatan, Baik, dan Sangat Baik.

Seluruh data merupakan data buatan untuk latihan, bukan data mahasiswa nyata.

## 5. Metode

Model yang digunakan adalah Decision Tree Classifier, yaitu algoritma Supervised Learning untuk klasifikasi.

Dataset dibagi menjadi 75% data latih dan 25% data uji. Pembagian dilakukan dengan stratifikasi agar proporsi kategori tetap terjaga.

Kinerja model dievaluasi menggunakan accuracy, precision, recall, F1-score, dan confusion matrix.

## 6. Hasil

Program menghasilkan evaluasi model, visualisasi Decision Tree, confusion matrix, dan prediksi kategori untuk satu data baru.

Hasil evaluasi perlu ditafsirkan secara hati-hati karena dataset berukuran kecil dan menggunakan label simulasi.

## 7. Cara Menjalankan

Instal library:

`python -m pip install pandas scikit-learn matplotlib`

Jalankan program:

`python prediksi_mahasiswa.py`

## 8. Output

- `hasil_evaluasi.png`: visualisasi evaluasi dan pohon keputusan.
- `hasil_prediksi.csv`: hasil prediksi data baru.

## 9. Kesimpulan

Proyek ini menunjukkan tahapan dasar Machine Learning, mulai dari pengolahan data, pelatihan model, evaluasi, hingga prediksi. Proyek ini digunakan untuk pembelajaran dan belum ditujukan untuk menilai performa mahasiswa dalam kondisi nyata.
