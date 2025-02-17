# Laporan Proyek Machine Learning - Ika Widiyanti

# Domain Proyek
  Stunting masih menjadi masalah kesehatan serius yang berdampak pada pertumbuhan fisik, perkembangan kognitif, serta daya tahan tubuh anak. Kondisi ini terjadi akibat kekurangan gizi dalam jangka panjang, terutama selama 1.000 hari pertama kehidupan. Faktor lain seperti keterbatasan akses pangan bergizi dan rendahnya pengetahuan ibu tentang nutrisi juga memperparah situasi (Masacgi & Rohman, 2023; Damayanti & Jakfar, 2023). Menurut WHO, stunting didefinisikan sebagai panjang atau tinggi badan yang lebih dari dua standar deviasi di bawah rata-rata usianya. Data tahun 2016 mencatat bahwa sekitar 22,9% balita di dunia mengalami stunting, dengan sekitar 3 juta kematian setiap tahun akibat kekurangan gizi (Muche et al., 2021). Di Indonesia, angka stunting sempat mencapai 38,9% pada tahun 2020. Meski turun menjadi 21,6% pada 2022, target nasional 14% di tahun 2024 masih cukup menantang (Kemenkes RI, 2022; Wahyuni & Kusumodestoni, 2024). Jika dibandingkan dengan negara tetangga seperti Malaysia (17%) dan Thailand (16%), Indonesia masih memiliki angka yang lebih tinggi, sehingga membutuhkan langkah konkret yang lebih efektif (Titimeidara & Hadikurniawati, 2021).
  
  Salah satu tantangan utama dalam menekan angka stunting adalah sulitnya mendeteksi risiko sejak dini. Stunting tidak hanya dipengaruhi oleh satu faktor, melainkan kombinasi dari usia, jenis kelamin, tinggi badan, kondisi ekonomi keluarga, serta pola asuh. Jika hanya mengandalkan pengamatan, tenaga kesehatan bisa kesulitan menentukan apakah seorang anak berisiko mengalami stunting. Oleh karena itu, penerapan teknologi seperti machine learning dapat menjadi solusi untuk memprediksi stunting dengan lebih akurat. Dengan memanfaatkan dataset Stunting Toddler (Balita) Detection yang berisi data dari 121.000 balita, model prediksi berbasis Random Forest dapat dikembangkan untuk mengidentifikasi anak yang berisiko mengalami stunting. Evaluasi model menggunakan metrik seperti akurasi, recall, heatmap, dan F1-score dapat membantu memastikan bahwa prediksi yang dihasilkan cukup akurat. Jika sistem ini diterapkan dengan baik, tenaga kesehatan dan pembuat kebijakan bisa lebih mudah menentukan strategi intervensi gizi yang tepat, sehingga anak-anak mendapatkan perhatian yang sesuai untuk mencegah stunting sejak dini.
  
Format Referensi : 

https://e-journal.hamzanwadi.ac.id/index.php/edumatic/article/view/27913

https://ejournal.pnc.ac.id/index.php/infotekmesin/article/view/2326

https://d1wqtxts1xzle7.cloudfront.net/81342538/537-libre.pdf?1645710345=&response-content-disposition=inline%3B+filename%3DAnalisis_Faktor_Faktor_Risiko_terhadap_K.pdf&Expires=1739725257&Signature=YweRJsDCwGRus-9IiXfUbn9-UPgfAyRbTNBRSw20cKOUOoiPSqdHLOF1pe-0PDy-2xJ6I7~bEnRETd27EjKJQd6Rh-qxob8kSBFcMeg~QE2R6OmwzH6PYt38Sa4lf8sKqYJ~cBG0OzcyyiIw8k9LJuMOvVdcuZaUwUUv8~PoTg4rMvPvXtzMlGLjgo9ZFT6Lc9l~9PAC1ZZQI11s7UBJ6KRvb7jc7RZmSI-NZ8xZe4lj-aNApZLXhGnviy0GDo4KPbtCWlpBjFHR4wZj3IYc-LmQYuvyOSldAIb7SHmGlOXpB1e1c5upY11HlOBAPIn~ZciR1gkJPLkPkgnUmlamZg__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA

# Business Understanding

# Problem Statement (Masalah yang ingin diselesaikan)
  Berdasarkan kondisi yang telah diuraikan sebelumnya, pada kasus ini akan mengembangkan sebuah sistem prediksi status gizi berdasarkan fitur yang ada untuk menjawab permasalahan berikut.
1. Bagaimana cara memprediksi status gizi balita berdasarkan usia, jenis kelamin, dan tinggi badan?
2. Bisakah kita mengembangkan model klasifikasi yang akurat untuk mendeteksi anak yang mengalami stunting?
3. Apakah ada pola tertentu antara usia, tinggi badan, dan risiko stunting?
# Goals (Tujuan)
  Untuk menjawab pertanyaan tersebut, proyek ini akan membuat predictive modelling dengan tujuan atau goals sebagai berikut:
1. Membangun model klasifikasi untuk memprediksi status gizi balita, yang termasuk kategori: "Severely Stunting", "Stunting", "Normal", "Tinggi".
2. Memberikan alat bantu untuk tenaga kesehatan dan pembuat kebijakan untuk mengidentifikasi balita yang berisiko mengalami stunting lebih awal.
3. Memahami faktor utama yang paling berpengaruh dalam menentukan status gizi.
# Metodologi
  Tujuan utama dari proyek ini adalah memprediksi status gizi. Status gizi merupakan variabel kategori karena memiliki tingkatan atau urutan, tetapi tidak berbentuk nilai numerik. Dalam predictive analytics, memprediksi variabel kategori berarti menyelesaikan permasalahan klasifikasi. Oleh karena itu, metodologi yang digunakan dalam proyek ini adalah membangun model klasifikasi dengan status gizi sebagai target.
# Metrik
  Evaluasi model dalam memprediksi status gizi dilakukan menggunakan berbagai metrik klasifikasi, termasuk akurasi, recall, F1-score, dan heatmap untuk menganalisis kesalahan prediksi secara lebih mendetail. Pengembangan model akan melibatkan beberapa algoritma machine learning, yaitu Random Forest dan Gradient Boosting. Model dengan tingkat kesalahan prediksi paling rendah akan dipilih sebagai model terbaik, dengan tujuan memperoleh prediksi yang seakurat mungkin.

# Data Understanding

  Pada proyek ini menggunakan dataset "Stunting Toddler (Balita) Detection"
Dataset ini merupakan kumpulan data berdasarkan rumus z-score penentuan stunting menurut WHO (World Health Organization), yang berfokus pada deteksi stunting pada balita (bayi dibawah lima tahun) yang dapat digunakan untuk membantu peneliti dan praktisi di bidang kesehatan dan gizi anak dalam perencanaan strategi pencegahan dan pengobatan stunting pada balita yang lebih baik. Dataset diambil dari Kaggle : https://www.kaggle.com/datasets/rendiputra/stunting-balita-detection-121k-rows
# Variabel-variabel pada dataset Stunting Toddler (Balita) Detection adalah sebagai berikut:
1. Umur (Bulan): Mengindikasikan usia balita dalam bulan. Rentang usia ini penting untuk menentukan fase pertumbuhan anak dan membandingkannya dengan standar pertumbuhan yang sehat. (Umur 0 sampai 60 bulan)
2. Jenis Kelamin: Terdapat dua kategori dalam kolom ini, 'laki-laki' (male) dan 'perempuan' (female). Jenis kelamin merupakan faktor penting dalam analisis pola pertumbuhan dan risiko stunting.
3. Tinggi Badan: Dicatat dalam centimeter, tinggi badan adalah indikator utama untuk menilai pertumbuhan fisik balita. Data ini memungkinkan peneliti untuk menentukan apakah pertumbuhan anak sesuai dengan standar usianya.
4. Status Gizi: Kolom ini dikategorikan menjadi 4 status - 'severely stunting', 'stunting', 'normal', dan 'tinggi'. 'Severely stunting' menunjukkan kondisi sangat serius (<-3 SD), 'stunting' menunjukkan kondisi stunting (-3 SD sd <-2 SD), 'normal' mengindikasikan status gizi yang sehat (-2 SD sd +3 SD), dan 'tinggi' (height) menunjukkan pertumbuhan di atas rata-rata (>+3 SD). Kategori ini membantu dalam identifikasi cepat dan intervensi bagi anak-anak yang berisiko atau mengalami masalah pertumbuhan.
# Tahapan yang Dilakukan
1. Mengimpor Library yang Dibutuhkan
Langkah pertama dalam penelitian ini adalah mengimpor seluruh library yang diperlukan. Beberapa library yang digunakan antara lain pandas, numpy, matplotlib, dan seaborn.
2. Dataset yang digunakan dalam penelitian ini memiliki format CSV dan dimuat menggunakan perintah pd.read_csv(url_dataset). Dataset tersebut terdiri dari 120.999 baris dan 4 kolom, yaitu umur, jenis kelamin, tinggi badan, dan status gizi.
3. Untuk melihat informasi keseluruhan dari dataset, digunakan fungsi DataFrame.info(). Fungsi ini menampilkan beberapa informasi penting, seperti:
- Jumlah kolom beserta nama dan tipe datanya
- Jumlah entri dalam dataset
- Memori yang digunakan
  
Berdasarkan hasil analisis, diketahui bahwa kolom jenis kelamin dan status gizi memiliki tipe data object, sedangkan kolom umur memiliki tipe data integer, dan tinggi badan memiliki tipe data float. Secara keseluruhan, dataset ini menggunakan memori sekitar 3,7 MB dengan RangeIndex: 120.999 entries (0 hingga 120.998).
4. Melihat Statistik Deskriptif Data
Analisis statistik deskriptif dilakukan menggunakan fungsi describe(), yang memberikan informasi statistik untuk setiap kolom numerik dalam dataset, termasuk:
- Count: jumlah sampel dalam dataset
- Mean: nilai rata-rata
  Std: standar deviasi
- Min: nilai minimum
- 25%: kuartil pertama (Q1)
- 50%: kuartil kedua atau median
- 75%: kuartil ketiga (Q3)
- Max: nilai maksimum
  
Karena hanya kolom umur dan tinggi badan yang bertipe numerik, maka analisis statistik ini diterapkan pada kedua kolom tersebut.
5. Mengecek Missing Value
Missing value merupakan data yang hilang atau kosong dalam sebuah dataset. Untuk memeriksa adanya missing value, digunakan fungsi isnull().sum(). Hasil analisis menunjukkan bahwa dataset tidak memiliki missing value, sehingga dapat langsung digunakan dalam proses pemodelan tanpa perlu dilakukan pembersihan data lebih lanjut.
6. Mengecek Outlier dengan Metode IQR
Outlier dideteksi menggunakan metode Interquartile Range (IQR), di mana:
- Q1 (Kuartil Pertama): 25% data berada di bawah nilai ini
- Q2 (Median/Kuartil Kedua): 50% data berada di bawah nilai ini
- Q3 (Kuartil Ketiga): 75% data berada di bawah nilai ini
IQR = Q3 - Q1

Untuk memvisualisasikan outlier, digunakan teknik boxplot. Boxplot memberikan gambaran mengenai penyebaran data, simetri, serta kemungkinan adanya outlier. Berdasarkan visualisasi boxplot, dapat disimpulkan bahwa kolom umur dan tinggi badan tidak memiliki outlier. Hal ini menunjukkan bahwa dataset telah bersih dan siap digunakan untuk tahap analisis lebih lanjut.

  
![Screenshot 2025-02-16 233858](https://github.com/user-attachments/assets/71e5e8cf-0476-492c-9093-11065aa9ee75)


![Screenshot 2025-02-16 233846](https://github.com/user-attachments/assets/dc489580-60e9-4dab-97bb-ca3339fc6c9d)



7. Mengecek Ukuran Dataset
Untuk memeriksa ukuran dataset, digunakan fungsi .shape(). Hasil yang diperoleh menunjukkan bahwa dataset terdiri dari 120.999 baris dan 4 kolom.
8. Analisis Data dengan Teknik Univariate EDA
Langkah pertama dalam analisis data menggunakan teknik Univariate Exploratory Data Analysis (EDA) adalah mengelompokkan variabel berdasarkan jenisnya, yaitu numerik dan kategorikal. Variabel numerik dalam dataset mencakup umur dan tinggi badan, sedangkan variabel kategorikal terdiri dari jenis kelamin dan status gizi.

a. Distribusi Status Gizi Berdasarkan Jenis Kelamin

Untuk mengidentifikasi pola tertentu antara status gizi dan jenis kelamin, dilakukan visualisasi menggunakan bar chart. Pada grafik ini, sumbu X (horizontal) merepresentasikan kategori status gizi, yang terdiri dari severely stunting, stunting, normal, dan tinggi. Sementara itu, sumbu Y (vertikal) menunjukkan jumlah individu dalam setiap kategori status gizi.

Jumlah individu dalam setiap kategori dihitung secara otomatis menggunakan sns.countplot(), yang kemudian ditampilkan dalam bentuk grafik. Selain itu, parameter hue= digunakan untuk membedakan jenis kelamin, di mana warna biru merepresentasikan laki-laki, sedangkan warna oranye merepresentasikan perempuan.



![Screenshot 2025-02-16 234530](https://github.com/user-attachments/assets/07d61030-4eec-494e-bc63-23522b73d091)




Hasil visualisasi ini adalah diperoleh

- Status gizi yang paling dominan adalah normal pada perempuan dibandingkan laki-laki, yang menunjukkan bahwa sebagian besar anak berada dalam kondisi gizi yang sehat (antara -2 SD hingga +3 SD).
- Status gizi "Severely Stunting" lebih banyak dibandingkan dengan "Stunting," dengan jumlah laki-laki sedikit lebih unggul dibandingkan perempuan. Kondisi ini perlu menjadi perhatian khusus tenaga kesehatan karena "Severely Stunting" menunjukkan kondisi yang sangat serius (<-3 SD) dan memerlukan penanganan segera.
- Status gizi "Tinggi" pada perempuan sedikit lebih unggul dibandingkan laki-laki. Kondisi ini menunjukkan pertumbuhan di atas rata-rata (>+3 SD).
- Status gizi "Stunting" menempati posisi terakhir dengan jumlah yang hampir setara antara perempuan dan laki-laki. Kondisi ini mengindikasikan stunting (-3 SD hingga <-2 SD) yang memerlukan perhatian lebih dari tenaga kesehatan dan menjadi fokus dalam penelitian lebih lanjut.
  
Kesimpulannya, sebagian besar balita menunjukkan pertumbuhan yang baik karena status gizi "Normal" mendominasi, meskipun terdapat sebagian kondisi balita yang memprihatinkan dalam kategori "Severely Stunting." Distribusi antara laki-laki dan perempuan hampir seimbang di setiap kategori, yang menunjukkan tidak ada perbedaan signifikan dalam status gizi berdasarkan jenis kelamin.
  
b. Pendistribusian fitur kategorikal yang kedua di kolom status gizi



![Screenshot 2025-02-16 234544](https://github.com/user-attachments/assets/e894cd2a-0452-4921-b109-3db35116a4aa)



Berdasarkan deskripsi variabel, urutan kategori status gizi dari yang tertinggi hingga terendah adalah normal, severely stunted, tinggi, dan stunting. Dari grafik ini, dapat disimpulkan bahwa kondisi gizi anak-anak bayi dan balita sebagian besar berada dalam kategori normal, yang menunjukkan pertumbuhan yang baik. Namun, perlu menjadi perhatian bahwa posisi kedua diisi oleh kategori severely stunted, yang menunjukkan kondisi gizi yang sangat buruk dan memerlukan perhatian khusus untuk penanganannya. Kondisi ini cukup tinggi di beberapa negara atau kota yang kurang mendapatkan perhatian, sehingga bayi dan balita mengalami penurunan gizi yang lebih parah, bahkan lebih buruk dibandingkan stunting. Selanjutnya, status gizi 'tinggi' berada di urutan ketiga, yang kemungkinan hanya terjadi pada sejumlah kecil anak yang mengalami pertumbuhan lebih cepat dari usianya. Terakhir, meskipun kategori stunting berada di urutan terakhir, tetap perlu mendapatkan perhatian agar kondisi gizi anak dapat diperbaiki dan kembali ke status normal.
  
c. Numerical Features

Berdasarkan grafik fitur numerik kolom 'umur' adalah:

- Ada peningkatan atau puncak diusia 60 bulan,artinya banyak bayi yang berada direntang usia 60 bulan.
- Hampir keseluruhan grafik merata dari umur 0 bulan atau yang baru lahir sampai umur 59 bulan,artinya usia balita tersebar secara merata.
- Namun ada penurunan juga di umur 10 bulan,dengan penurunan yang cukup banyak tetapi kemudian naik lagi.



![Screenshot 2025-02-16 234554](https://github.com/user-attachments/assets/47364818-d4eb-46c5-8a0f-473f7f53572f)



  
Distribusi fitur numerik kolom 'Tinggi Badan'

- Mayoritas balita memiliki tinggi 90-100 cm.
- Distribusi tinggi badan simetris—tidak ada skewness yang signifikan.
- Ada sedikit balita dengan tinggi di bawah 50 cm atau di atas 120 cm.



![Screenshot 2025-02-16 234603](https://github.com/user-attachments/assets/b4706376-0fd7-4b10-9c86-0a86aa692dd9)



  
# Data Preparation

Pada tahapan data preparation,proyek ini menerapkan tiga teknik yaitu sebagai berikut:

1. Encoding Fitur Kategori

   Untuk encoding fitur kategori, salah satu teknik yang umum digunakan adalah label encoding. Mengingat tujuan dari prediksi adalah status gizi, yang merupakan variabel kategori dengan tingkatan/urutan, teknik label encoding sangat cocok digunakan. Library scikit-learn menyediakan fungsi untuk menghasilkan fitur baru yang sesuai, sehingga variabel kategori dapat terwakili dengan baik. Dalam dataset ini, terdapat tiga variabel kategori, yaitu 'Jenis Kelamin' dan 'Status Gizi'. Setelah diterapkan, kolom 'Jenis Kelamin' menghasilkan output 0 untuk laki-laki dan 1 untuk perempuan. Begitu pula dengan kolom 'Status Gizi', yang telah diubah menjadi variabel numerik.
   
   
Kode: 


![Screenshot 2025-02-16 235248](https://github.com/user-attachments/assets/b2879a79-4e8b-4b0f-b2b7-c2f5b79021dc)



Output: 


![Screenshot 2025-02-16 235306](https://github.com/user-attachments/assets/2d476292-ea51-4669-b93d-9b635df4070e)




2. Pembagian dataset dengan fungsi train_test_split dari library sklearn.

   Pembagian dataset menjadi data latih (training) dan data uji (testing) merupakan langkah yang penting sebelum membangun model. Tujuan dari pembagian ini adalah untuk menghindari kebocoran informasi dari data latih ke data uji. Umumnya, proporsi pembagian antara data latih dan data uji adalah 80:20. Proses pembagian ini dapat dilakukan menggunakan fungsi train_test_split dengan kode berikut.
   
   
Kode:


![Screenshot 2025-02-16 235515](https://github.com/user-attachments/assets/e63a2e97-2f3c-4974-997f-ffc35d9b7c66)



Output: 


![Screenshot 2025-02-16 235648](https://github.com/user-attachments/assets/cabe63a4-89bf-405f-bc35-b5cd618198ae)



3. Standarisasi

   Standarisasi adalah teknik transformasi yang umum digunakan dalam tahap persiapan pemodelan. Untuk fitur numerik, transformasi tidak dilakukan menggunakan one-hot encoding seperti pada fitur kategori. Sebagai gantinya, digunakan teknik StandardScaler dari library Scikit-learn. StandardScaler melakukan proses standarisasi fitur dengan mengurangi nilai rata-rata (mean) dan membaginya dengan standar deviasi (standard deviation), yang menggeser distribusi data. Hasil dari proses ini adalah distribusi dengan standar deviasi sebesar 1 dan mean sebesar 0. Sekitar 68% dari nilai akan berada dalam rentang -1 hingga 1. Fitur standarisasi diterapkan pada data pelatihan (train) dan data uji (test) untuk kolom numerik seperti 'umur' dan 'tinggi badan'. Setelah transformasi, outputnya memiliki nilai mean = 0 dan standar deviasi = 1.

   
kode: 


![Screenshot 2025-02-16 235613](https://github.com/user-attachments/assets/e34633f8-777a-41ba-b662-825d784382d8)

Output:


![Screenshot 2025-02-16 235632](https://github.com/user-attachments/assets/862e9ec1-c3d1-40f8-8197-501175ab41de)



# Modeling

  Pada tahap ini menggunakan algoritma Random Forest untuk membangun model klasifikasi yang dapat memprediksi status gizi balita berdasarkan fitur-fitur seperti usia, jenis kelamin, dan tinggi badan. Berikut adalah penjelasan parameter yang digunakan dan cara kerja algoritma:
# Parameter yang Digunakan 
1. n_estimators=50:
Jumlah pohon keputusan (decision trees) yang akan dibangun dalam model Random Forest. Semakin banyak pohon, semakin baik model dalam menangkap pola data, tetapi juga membutuhkan waktu pelatihan yang lebih lama.
2. max_depth=16:
Kedalaman maksimum setiap pohon keputusan. Parameter ini membatasi seberapa dalam pohon dapat tumbuh. Jika terlalu dalam, model bisa overfitting; jika terlalu dangkal, model bisa underfitting
3. random_state=55:
Digunakan untuk memastikan hasil yang konsisten setiap kali model dijalankan. Ini penting untuk reproduktibilitas.
4. n_jobs=-1:
Menggunakan semua core CPU yang tersedia untuk mempercepat proses pelatihan model.
# Cara Kerja Random Forest:
Random Forest adalah algoritma ensemble learning yang menggabungkan banyak pohon keputusan (decision trees) untuk membuat prediksi yang lebih akurat dan stabil. Secara umum, Random Forest bekerja dengan cara menggabungkan banyak pohon keputusan yang membagi data berdasarkan fitur-fitur seperti usia, jenis kelamin, dan tinggi badan untuk memprediksi status gizi. Model ini memanfaatkan prinsip bagging (Bootstrap Aggregating), di mana setiap pohon dibangun dengan data latih yang diambil secara acak dengan pengembalian (bootstrap). Hasil dari banyak pohon keputusan akan digabungkan untuk menghasilkan prediksi yang lebih stabil dan akurat.
# Kelebihan
1. Mampu menangani data numerik dan kategorikal.
2. Tidak rentan terhadap overfitting karena menggunakan banyak pohon keputusan.
3. Dapat mengukur pentingnya setiap fitur dalam membuat keputusan.
# Kekurangan
1. Proses pelatihan yang memakan waktu lebih lama jika dibandingkan dengan model lain seperti regresi logistik.
2. Kurang mudah untuk diinterpretasi dibandingkan dengan pohon keputusan tunggal yang lebih sederhana.
   
# Evaluation 

Pada tahap evaluasi,menggunakan beberapa metrik untuk mengukur performa model:
1. Accuracy:
Akurasi mengukur seberapa sering model memprediksi dengan benar. Pada proyek ini, akurasi model adalah 99.91%, yang menunjukkan model sangat akurat.
2. Classification Report
- Precision: Mengukur seberapa akurat prediksi positif model.
- Recall: Mengukur seberapa banyak kasus positif yang berhasil dideteksi.
- F1-Score: Merupakan rata-rata harmonik dari precision dan recall.
- Hasil Precision, recall, dan F1-score untuk semua kelas adalah 1.00 (100%), yang menunjukkan model sangat baik dalam memprediksi semua kategori status gizi.
3. Confusion Matrix
Penjelasan:
Confusion matrix menunjukkan jumlah prediksi yang benar dan salah untuk setiap kelas. Diagonal utama menunjukkan prediksi yang benar.
Hasil:
Dari visualisasi, terlihat bahwa hampir semua prediksi benar (angka-angka di diagonal utama sangat tinggi).
4. Feature Importance:
Penjelasan:
Feature importance menunjukkan seberapa penting setiap fitur dalam memprediksi status gizi.
Hasil:
Tinggi Badan (cm) memiliki importance tertinggi, diikuti oleh Umur (bulan) dan Jenis Kelamin. Ini menunjukkan bahwa tinggi badan adalah faktor paling penting dalam memprediksi stunting.
# Dampak terhadap Business Understanding
Model yang dibangun berhasil menjawab problem statement yang diajukan:
1. Model ini berhasil memprediksi status gizi balita dengan akurasi yang sangat tinggi, terutama dalam mengidentifikasi anak-anak yang berisiko mengalami stunting.
2. Dengan analisis feature importance, dapat disimpulkan bahwa tinggi badan, umur, dan jenis kelamin adalah faktor penting yang harus diperhatikan dalam memonitor status gizi balita, yang sesuai dengan kebutuhan bisnis untuk memfokuskan intervensi gizi pada kelompok rentan tersebut.
3. Dengan hasil yang sangat baik ini, model ini bisa digunakan dalam sistem prediksi untuk mendeteksi anak-anak yang berisiko stunting, memberikan informasi yang berguna bagi pengambil kebijakan dan pihak kesehatan dalam merancang program perbaikan gizi.
