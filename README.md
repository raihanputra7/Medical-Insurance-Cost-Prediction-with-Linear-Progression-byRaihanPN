# Prediksi Biaya Asuransi Kesehatan dengan Machine Learning

Halo, saya **Raihan Putra Nurhamsyah**, mahasiswa Teknik Informatika Universitas Pamulang. Selamat datang di repositori tugas praktikum machine learning saya! 👋

Di repositori ini saya membangun model untuk **memprediksi biaya asuransi kesehatan** seseorang berdasarkan data pribadinya, seperti usia, BMI, status perokok, dan wilayah tempat tinggal. Seluruh prosesnya saya kerjakan dari nol sampai model jadi, mengikuti alur yang diajarkan dosen saya: mulai dari mengumpulkan data, membersihkan, menganalisis, membagi data, melatih model, sampai mengevaluasi dan menyimpannya.

README ini saya tulis supaya siapa pun yang membuka repositori ini, bahkan yang belum pernah belajar machine learning, bisa paham **apa yang saya kerjakan dan kenapa setiap langkahnya perlu dilakukan**.

---

## Sebenarnya, ini proyek tentang apa?

Bayangkan sebuah perusahaan asuransi. Setiap calon nasabah punya kondisi yang berbeda-beda, dan biaya premi atau klaimnya pun berbeda. Pertanyaannya: **faktor apa yang paling memengaruhi biaya itu, dan bisakah kita memprediksinya?**

Itulah yang saya jawab di proyek ini. Karena yang diprediksi adalah angka (biaya dalam dolar), jenis masalahnya adalah **regresi** pada *supervised learning*. Kalau targetnya berupa kategori, misalnya kelas biaya "rendah, sedang, tinggi", maka algoritma yang dipakai berganti menjadi klasifikasi.

## Dataset yang dipakai

Saya memakai **Medical Cost Personal Datasets** (`insurance.csv`), dataset terbuka yang tersedia di Kaggle dan GitHub. Isinya 1.338 baris data dengan 7 kolom:

| Kolom | Penjelasan |
|---|---|
| `age` | Usia penerima manfaat utama |
| `sex` | Jenis kelamin (female atau male) |
| `bmi` | Body mass index, ukuran proporsi berat dan tinggi badan |
| `children` | Jumlah anak atau tanggungan yang dicakup asuransi |
| `smoker` | Apakah perokok atau bukan (yes atau no) |
| `region` | Wilayah tinggal di AS (northeast, northwest, southeast, southwest) |
| `charges` | **Biaya asuransi, ini target yang diprediksi** |

---

## Perjalanan saya, tahap demi tahap

Berikut urutan yang saya kerjakan di notebook. Di setiap tahap saya jelaskan **apa yang dilakukan** dan **kenapa harus dilakukan**.

### 1. Data Collecting dan Data Loading
Saya mengambil dataset dari sumber terbuka (lewat tautan GitHub) menggunakan `pandas`, dengan file lokal sebagai cadangan jika internet bermasalah. Setelah itu saya membuat salinan data kerja `df`.

**Kenapa begitu?** Semua proyek machine learning dimulai dari data. Salinan dibuat supaya data asli tetap utuh. Kalau suatu langkah pembersihan salah, saya tinggal mengulang dari data asli.

### 2. Data Cleaning (membersihkan data)
Data mentah hampir selalu "kotor", dan model yang dilatih dari data kotor akan memberi hasil yang kurang bisa dipercaya. Ada empat hal yang saya periksa:

- **Informasi dan statistik dataset** (`info()` dan `describe()`): memastikan tipe data benar dan melihat sebaran awal angka. Dari sini terlihat biaya (`charges`) sangat condong ke kanan, artinya sebagian besar biaya kecil dan hanya sedikit yang sangat besar.
- **Missing value** (data kosong): saya memeriksa dan menyiapkan penanganannya (numerik diisi median, kategorikal diisi modus, kolom yang kosongnya lebih dari 75% dibuang). Pada dataset ini ternyata **tidak ada data kosong**, tetapi kodenya tetap saya jalankan agar alurnya lengkap dan bisa dipakai ulang di dataset lain.
- **Outlier** (nilai ekstrem): saya mendeteksi dengan box plot dan metode IQR. Hasilnya, 9 baris dengan BMI sangat tinggi saya hapus. **Saya sengaja tidak menghapus outlier pada kolom `charges`**, karena biaya yang tinggi itu data asli milik perokok, bukan kesalahan input. Kalau dihapus (sekitar 10% data), model akan kehilangan pola terpenting yang justru harus ia pelajari.
- **Data duplikat**: ada 1 baris ganda yang saya hapus supaya satu pengamatan tidak dihitung dua kali.

### 3. Data Transformation (mengubah data agar siap dipakai model)
- **Standardisasi** (`StandardScaler`): fitur numerik seperti `age`, `bmi`, dan `children` diubah agar rata-ratanya 0 dan simpangan bakunya 1. *Kenapa?* Skala antarfitur berbeda jauh (usia puluhan, jumlah anak satuan), dan model regresi lebih stabil bila skalanya seragam. Kolom target `charges` sengaja tidak diskalakan supaya nilai error tetap dalam dolar dan mudah dipahami.
- **Encoding**: komputer hanya mengerti angka, jadi kolom teks harus diubah. Saya membandingkan dua cara, yaitu *One-Hot Encoding* dan *Label Encoding*. Hasil akhirnya: `sex` dan `smoker` (hanya dua pilihan) memakai Label Encoding, sedangkan `region` (empat wilayah tanpa urutan) memakai One-Hot Encoding. *Kenapa dibedakan?* Kalau wilayah diberi angka 0, 1, 2, 3, model bisa salah mengira ada "urutan" antarwilayah.

Hasil akhir tahap ini adalah dataset final berukuran **1.328 baris × 10 kolom**.

### 4. EDA dan ExDA (mengenali data lewat grafik)
Sebelum melatih model, saya "berkenalan" dulu dengan data lewat visualisasi: distribusi biaya, perbandingan biaya antarwilayah, hubungan biaya dengan usia dan BMI, serta matriks korelasi.

**Kenapa penting?** Dari tahap ini saya sudah bisa menebak jawabannya sebelum model dibuat: **status perokok adalah faktor yang paling berpengaruh** (korelasi sekitar 0,79 terhadap biaya), disusul usia dan BMI. Ada juga temuan menarik: pada perokok, biaya naik tajam ketika BMI naik, sedangkan pada bukan perokok hampir tidak berpengaruh. Artinya ada *interaksi* antarfitur yang nanti menentukan model mana yang cocok.

### 5. Data Splitting (membagi data)
Data dibagi **80% untuk latih** (1.062 baris) dan **20% untuk uji** (266 baris) dengan `train_test_split`.

**Kenapa harus dibagi?** Ibarat belajar untuk ujian: kalau soal ujian sama persis dengan soal latihan, nilai bagus belum tentu berarti paham. Data uji adalah "soal ujian" yang belum pernah dilihat model, sehingga kemampuan model yang sebenarnya bisa dinilai secara jujur.

### 6. Modelling (membangun model)
Saya fokus pada **dua model** agar perbandingannya jelas:

- **Linear Regression**: model sederhana yang mencari garis lurus antara fitur dan biaya. Ia berperan sebagai *baseline* (pembanding dasar).
- **Gradient Boosting Regressor**: model yang membangun banyak pohon keputusan secara berurutan, di mana setiap pohon memperbaiki kesalahan pohon sebelumnya. Model ini mampu menangkap hubungan yang tidak lurus.

**Kenapa dua model?** Dengan membandingkan model sederhana dan model kompleks, kita bisa melihat apakah kerumitan tambahan itu benar-benar memberi hasil yang lebih baik. Dasar ilmiah pemilihan algoritma ini, lengkap dengan rujukan jurnal, ada di bagian 7c.

### 7. Evaluasi Model
Kedua model diuji pada data uji dengan empat ukuran: **MAE** (rata-rata selisih prediksi dan nilai sebenarnya), **MSE**, **RMSE** (satuannya dolar), dan **R²** (seberapa besar variasi biaya yang bisa dijelaskan model, makin dekat ke 1 makin baik).

| Model | MAE | RMSE | R² (data uji) | R² (data latih) |
|---|---|---|---|---|
| **Gradient Boosting** | 2.559 | 4.510 | **0,876** | 0,899 |
| Linear Regression | 4.355 | 6.105 | 0,772 | 0,742 |

**Cara membacanya:** Gradient Boosting jauh lebih baik, dengan rata-rata kesalahan hampir separuh dari Linear Regression. R² data latih dan data ujinya juga berdekatan, jadi model tidak sekadar menghafal (tidak *overfitting*). Linear Regression kalah karena hubungan antara fitur dan biaya memang tidak lurus, terutama efek perokok yang berinteraksi dengan BMI dan usia.

Saya juga menambahkan grafik *feature importance* dan *plot residual*. Menurut Gradient Boosting, `smoker` menyumbang sekitar 68% pengaruh, `bmi` sekitar 18%, dan `age` sekitar 12%. Hasil ini konsisten dengan temuan di tahap EDA.

#### 7a. Kenapa saya memakai MAE, MSE/RMSE, dan R²? (rujukan jurnal)

Dosen saya meminta agar pemilihan ukuran evaluasi ini didukung referensi ilmiah, bukan sekadar ikut-ikutan. Karena itu saya menelusuri lima jurnal tentang prediksi biaya asuransi kesehatan yang memakai dataset yang sama atau sangat mirip (Medical Cost Personal Datasets, 1.338 data), lalu merangkum apa kata jurnal-jurnal itu tentang MAE, MSE, RMSE, dan R². Daftar lengkapnya ada di bagian [Referensi](#referensi) di bawah, dan angka [1] sampai [5] di sini merujuk ke daftar tersebut.

**MAE (Mean Absolute Error)** adalah rata-rata selisih mutlak antara nilai prediksi dan nilai aktual, dengan rumus $MAE = \frac{1}{n}\sum |y_i - \hat{y}_i|$. Dwikasari dkk. [1] menjelaskan bahwa MAE dan MSE menggambarkan perbedaan antara nilai prediksi dan nilai aktual biaya medis. Hesananda dkk. [3] menafsirkan MAE sebagai galat rata-rata yang diharapkan pada prediksi biaya individu, dan menetapkan target praktis **MAE ≤ 3.000** agar galat tetap dalam kisaran yang operasional. Fauzi dkk. [2] memakai MAE bersama MSE dan MAPE untuk memilih model dengan error terkecil. Kelebihan MAE, satuannya sama dengan target (dolar), jadi mudah dibayangkan: MAE 2.559 berarti rata-rata prediksi meleset sekitar 2.559 dolar.

**MSE (Mean Squared Error) dan RMSE** menghitung kuadrat selisih, dengan rumus $MSE = \frac{1}{n}\sum (y_i - \hat{y}_i)^2$ dan $RMSE = \sqrt{MSE}$. Karena selisih dikuadratkan, kesalahan besar dihukum lebih berat daripada kesalahan kecil. Hesananda dkk. [3] menyebut RMSE memberi penalti lebih besar pada galat besar sehingga sensitif terhadap ekor distribusi `charges`, yaitu kasus mahal yang jarang tetapi penting secara finansial. Parlika dkk. [5] juga menandai RMSE sebagai ukuran yang sensitif terhadap outlier. Al Haddad dkk. [4] memakai RMSE sebagai ukuran evaluasi utama (sekitar 5.891 sampai 5.903), sedangkan Fauzi dkk. [2] memakai MSE untuk membandingkan model ANN dengan Multiple Linear Regression. Nilai MSE bersatuan dolar kuadrat sehingga angkanya sangat besar, itulah sebabnya RMSE lebih enak dibaca.

**R² (koefisien determinasi)** menunjukkan seberapa besar variasi biaya yang bisa dijelaskan model, dengan rumus $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$. Dwikasari dkk. [1] memakai R² untuk mengukur sejauh mana variasi variabel dependen (biaya medis) dapat dijelaskan oleh variabel independen. Hasil mereka menarik: Multiple Linear Regression mencapai R² **79,99%**, sedangkan Simple Linear Regression yang hanya memakai usia cuma **7,86%**. Hesananda dkk. [3] menjelaskan R² sebagai proporsi variasi yang dijelaskan model dan menetapkan target **R² ≥ 0,80**. Al Haddad dkk. [4] melaporkan squared correlation 0,81, dan Parlika dkk. [5] merujuk karya Chicco dkk. (2021) yang menyatakan bahwa R² lebih informatif dibanding SMAPE, MAE, MAPE, MSE, dan RMSE.

| Ukuran | Cara membaca | Rujukan |
|---|---|---|
| MAE | Rata-rata meleset berapa dolar. Makin kecil makin baik. | [1], [2], [3], [5] |
| MSE | Rata-rata kuadrat selisih. Makin kecil makin baik, tetapi satuannya dolar kuadrat. | [1], [2] |
| RMSE | Akar MSE, satuannya dolar. Peka terhadap kesalahan besar dan outlier. | [3], [4], [5] |
| R² | Proporsi variasi biaya yang dijelaskan model. Makin dekat ke 1 makin baik. | [1], [3], [4] |

**Kenapa tidak memakai MAPE?** Fauzi dkk. [2] dan Parlika dkk. [5] memakai MAPE (error dalam persen). Parlika dkk. sendiri mencatat kelemahannya: MAPE tidak bisa dihitung bila nilai aktual nol, cenderung bias pada angka yang sangat kecil, dan rentang nilainya bisa mencapai tak hingga. Saya memilih MAE, RMSE, dan R² (ditambah MSE) karena kombinasi inilah yang dipakai Hesananda dkk. [3], sehingga hasil saya bisa dibandingkan langsung dengan target yang mereka tetapkan.

#### 7b. Hasil saya dibandingkan dengan jurnal

| Sumber | Model | MAE | RMSE | R² |
|---|---|---|---|---|
| Dwikasari dkk. [1] | Multiple Linear Regression | - | - | 0,7999 |
| Hesananda dkk. [3] | Linear Regression | 3.449,75 | 5.937,03 | ± 0,74 |
| Hesananda dkk. [3] | Random Forest | 2.521,98 | 4.548,38 | 0,8667 |
| Al Haddad dkk. [4] | Multiple Linear Regression | - | ± 5.891 sampai 5.903 | 0,81 (squared correlation) |
| Parlika dkk. [5] | Linear Regression | 2.994,60 | 5.693,95 | - (MAPE 18,48%) |
| Fauzi dkk. [2] | Multiple Linear Regression | 3.769,42 | ± 5.459 (dihitung dari MSE) | - |
| Fauzi dkk. [2] | ANN | 1.242,59 | ± 3.964 (dihitung dari MSE) | - |
| **Saya** | **Linear Regression** | **4.355,38** | **6.104,72** | **0,7723** |
| **Saya** | **Gradient Boosting** | **2.559,39** | **4.509,57** | **0,8757** |

Beberapa hal yang bisa dibaca dari tabel di atas. Pertama, Linear Regression saya (R² 0,77) berada di rentang model linear pada jurnal (R² sekitar 0,74 sampai 0,81), walaupun MAE dan RMSE saya sedikit lebih tinggi. Perbedaan seperti ini wajar karena pembagian data, praktik preprocessing, dan jumlah fitur yang dipakai tiap penelitian tidak sama (Fauzi dkk. [2], misalnya, memakai fitur tambahan berupa jumlah langkah dan riwayat klaim, dan Parlika dkk. [5] membuang kolom `region`). Penyebab pasti selisihnya belum saya uji. Kedua, Gradient Boosting saya (R² 0,876, MAE 2.559) sangat dekat dengan Random Forest pada Hesananda dkk. [3] (R² 0,8667, MAE 2.522), yaitu sama-sama model ensemble berbasis pohon. Ketiga, dengan target yang ditetapkan Hesananda dkk. [3], Gradient Boosting saya **memenuhi kedua target** (R² ≥ 0,80 dan MAE ≤ 3.000), sedangkan Linear Regression saya belum memenuhi keduanya.

#### 7c. Kenapa algoritmanya Linear Regression (dan kenapa ditambah Gradient Boosting)?

Dosen saya juga meminta dasar penentuan algoritma. Jawabannya saya susun dari kelima jurnal tadi.

**Jenis masalahnya regresi.** Yang diprediksi adalah biaya (angka kontinu), dan kelima jurnal memperlakukannya sebagai tugas estimasi atau prediksi numerik. Dwikasari dkk. [1] menjelaskan regresi linear sebagai teknik statistik untuk memodelkan hubungan linear antara variabel dependen dan variabel independen, dengan persamaan $Y = B_0 + B_1X_1 + B_2X_2 + \dots + B_nX_n$ untuk kasus banyak variabel. Al Haddad dkk. [4] memakai regresi linear berganda karena melibatkan beberapa variabel independen (usia, jenis kelamin, berat badan, perokok, wilayah) terhadap satu variabel dependen, yaitu biaya asuransi.

**Jenis metodenya Multiple Linear Regression.** Perbandingan pada Dwikasari dkk. [1] menjadi alasan utama: model yang hanya memakai satu fitur (usia) hanya menjelaskan 7,86% variasi biaya, sedangkan model yang memakai semua fitur menjelaskan 79,99%. Itulah sebabnya Linear Regression pada notebook saya memakai seluruh fitur sekaligus, bukan satu fitur saja. Kelebihan lainnya adalah interpretabilitas. Hesananda dkk. [3] menyebut Linear Regression mudah diinterpretasi, dan Parlika dkk. [5] menyebut Multiple Linear Regression unggul dalam interpretabilitas, berfungsi sebagai baseline yang solid, serta termasuk model transparan (white-box). Karena itu Linear Regression saya jadikan model dasar, dan koefisiennya bisa dibaca langsung pada tabel koefisien di notebook.

**Kenapa ditambah Gradient Boosting.** Jurnal-jurnal itu juga menunjukkan keterbatasan model linear. Hesananda dkk. [3] mencatat bahwa distribusi `charges` miring ke kanan dan efek `smoker` yang makin besar pada usia tertentu sulit ditangkap model linear sederhana, dan hasilnya Linear dan Ridge hanya mencapai R² sekitar 0,74 sedangkan Random Forest mencapai 0,8667. Fauzi dkk. [2] menemukan bahwa ANN mengungguli Multiple Linear Regression (MAE 1.242,59 berbanding 3.769,42) terutama dalam menangani pola kompleks dan interaksi non-linear antarvariabel. Parlika dkk. [5] menyebut Multiple Linear Regression sering kalah akurat dibanding model yang lebih canggih seperti Gradient Boosting dalam menangani hubungan non-linear. Pola ini cocok dengan temuan EDA saya, yaitu efek perokok yang berinteraksi dengan BMI. Maka saya membandingkan Linear Regression (sederhana dan mudah dibaca) dengan Gradient Boosting (kompleks dan menangkap non-linearitas), dan hasil saya mendukung temuan jurnal tersebut.

Satu catatan agar jujur: dari kelima jurnal, tidak ada yang menguji Gradient Boosting secara langsung pada dataset ini. Yang diuji adalah Random Forest [3] dan ANN [2]. Gradient Boosting saya pilih karena diajarkan pada modul dosen, satu keluarga dengan Random Forest (sama-sama ensemble pohon), dan disebut Parlika dkk. [5] sebagai model yang lebih baik untuk hubungan non-linear.

**Selaras dengan alur di jurnal.** Pembagian data 80:20 yang saya pakai sama dengan Dwikasari dkk. [1], Fauzi dkk. [2], dan Parlika dkk. [5], sedangkan Al Haddad dkk. [4] mencoba 9:1, 8:2, dan 7:3. Korelasi hasil EDA saya (`smoker` 0,79, `age` 0,30, `bmi` 0,19) juga sama dengan yang dilaporkan Al Haddad dkk. [4], dan fitur terpenting menurut Gradient Boosting (`smoker`, `bmi`, `age`) sama dengan feature importance Random Forest pada Hesananda dkk. [3].

### 8. Penyimpanan Data dan Model
Dataset bersih dan model terbaik disimpan memakai `joblib` agar bisa dipakai lagi tanpa mengulang semua proses. `scaler` dan urutan kolom fitur juga ikut disimpan, karena data baru harus diproses dengan cara yang **persis sama** seperti saat model dilatih.

---

## Kesimpulan

Faktor yang paling memengaruhi biaya asuransi pada data ini adalah **status perokok**, lalu **usia** dan **BMI**. Model terbaik adalah **Gradient Boosting Regressor** dengan R² sekitar **0,88** pada data uji. Pelajaran terpentingnya: sebelum memilih model, pahami dulu datanya, karena pola yang saya temukan lewat EDA (efek perokok dan interaksinya dengan BMI) langsung menjelaskan kenapa model non-linear lebih unggul.

---

## Isi repositori

```
.
├── prediksi-biaya-asuransi.ipynb   # notebook utama (kode + penjelasan tiap langkah)
├── README.md                       # dokumen ini
└── (hasil setelah notebook dijalankan)
    ├── insurance_clean.csv         # dataset final hasil pemrosesan
    ├── gbr_model.joblib            # model Gradient Boosting yang sudah dilatih
    ├── scaler.joblib               # scaler untuk standardisasi data baru
    └── fitur_kolom.joblib          # urutan kolom fitur
```

Setiap cell kode di notebook diawali penjelasan singkat dalam bahasa Indonesia, jadi bisa dibaca seperti modul belajar.

## Cara menjalankan

**Opsi 1: Google Colab (paling mudah).** Unggah file `prediksi-biaya-asuransi.ipynb` ke [Google Colab](https://colab.research.google.com), lalu pilih *Runtime → Run all*. Data akan diunduh otomatis dari internet.

**Opsi 2: di komputer sendiri.**

```bash
# 1. Salin repositori
git clone <URL-REPOSITORI-INI>
cd <NAMA-FOLDER>

# 2. Pasang pustaka yang dibutuhkan
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter

# 3. Buka notebook
jupyter notebook prediksi-biaya-asuransi.ipynb
```

## Teknologi yang dipakai

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, dan joblib.

## Catatan dan rencana pengembangan

Saya menulis ini apa adanya supaya jujur soal batasannya:

- Standardisasi saat ini dilakukan **sebelum** pembagian data, mengikuti urutan materi dosen. Secara teori ini memungkinkan sedikit kebocoran informasi dari data uji. Perbaikan idealnya: `scaler` dilatih hanya pada data latih, misalnya memakai `Pipeline`. Saran memakai pipeline untuk mencegah kebocoran data ini juga disebut Hesananda dkk. [3].
- Evaluasi masih memakai satu kali pembagian data. Ke depan saya ingin mencoba *cross validation* dan *hyperparameter tuning*. Hesananda dkk. [3] juga merekomendasikan k-fold cross-validation untuk menilai kestabilan performa.
- Dataset ini kecil dan tidak memuat faktor seperti riwayat penyakit, sehingga masih ada kasus yang prediksinya meleset jauh.

## Referensi

Seluruh rujukan pada bagian Evaluasi Model (7a sampai 7c) bersumber dari lima jurnal berikut.

1. Dwikasari, N. M. D., Sutramiani, N. P., Putri, K. S. Y., Kusuma, N. T. R., Pramana, M. D. A. D., & Darma, I. W. A. S. (2023). Medical Costs Estimation Using Linear Regression Method. *Jurnal Ilmiah Merpati*, 11(3), 171-179. https://doi.org/10.24843/JIM.2023.v11.i03.p03
2. Fauzi, R., Nabila, A. C., Krisnawati, V. F., & Rifai, A. P. (2024). Prediksi Premi Asuransi Kesehatan Menggunakan Machine Learning: Pendekatan Artificial Neural Network. *Journal of Industrial Engineering and Operation Management (JIEOM)*, 7(1), 52-63. https://doi.org/10.31602/jieom.v7i1.13802
3. Hesananda, R., Fasya, A., & Zulfariansyah, M. *Analisis Prediksi Biaya Asuransi Kesehatan Menggunakan Metode Machine Learning*, hlm. 123-131, e-ISSN 2621-5934.
4. Al Haddad, B., Bahtiar, A., & Dwilestari, G. (2024). Implementasi Algoritma Regresi Linear Berganda untuk Memprediksi Biaya Asuransi Kesehatan. *Jurnal Informatika dan Rekayasa Perangkat Lunak*, 6(1), 142-149.
5. Parlika, R., Ramadhana, F., Muslem, I., & Afriana. (2026). Implementasi Algoritma Linear Regression pada Framework Laravel untuk Prediksi Biaya Asuransi Kesehatan. *NOVA Komputa*, 1(1). https://jurnal.gsminnovation.com/index.php/novakomputa/article/view/1

---

## Sumber data

Choi, M. *Medical Cost Personal Datasets*, Kaggle: <https://www.kaggle.com/datasets/mirichoi0218/insurance> (data asli berasal dari buku *Machine Learning with R* karya Brett Lantz).

---

Terima kasih sudah mampir dan membaca sampai akhir! Kalau ada saran atau masukan, silakan buka *issue* di repositori ini. 🙌

**Raihan Putra Nurhamsyah**
Teknik Informatika, Universitas Pamulang
