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
  Berdasarkan kondisi yang telah diuraikan sebelumnya, pada kasus ini kita akan akan mengembangkan sebuah sistem prediksi status gizi berdasarkan fitur yang ada untuk menjawab permasalahan berikut.
1. Bagaimana cara memprediksi status gizi balita berdasarkan usia, jenis kelamin, dan tinggi badan?
2. Bisakah kita mengembangkan model klasifikasi yang akurat untuk mendeteksi anak yang mengalami stunting?
3. Apakah ada pola tertentu antara usia, tinggi badan, dan risiko stunting?
# Goals (Tujuan)
  Untuk menjawab pertanyaan tersebut, Anda akan membuat predictive modelling dengan tujuan atau goals sebagai berikut:
1. Membangun model klasifikasi untuk memprediksi status gizi balita, yang termasuk kategori: "Severely Stunting", "Stunting", "Normal", "Tinggi".
2. Memberikan alat bantu untuk tenaga kesehatan dan pembuat kebijakan untuk mengidentifikasi balita yang berisiko mengalami stunting lebih awal.
3. Memahami faktor utama yang paling berpengaruh dalam menentukan status gizi.
# Metodologi
  Prediksi status gizi adalah tujuan yang ingin dicapai. Seperti yang kita ketahui,status gizi adalah variabel kategori karena status gizi punya tingkatan/urutan tetapi tidak berupa nilai. Dalam predictive analytics, saat membuat prediksi variabel kategori artinya kita sedang menyelesaikan permasalahan klasifikasi. Oleh karena itu, metodologi pada proyek ini adalah: membangun model klasifikasi dengan status gizi sebagai target.
# Metrik
  Metrik digunakan untuk mengevaluasi seberapa baik model dalam memprediksi status gizi. Untuk kasus klasifikasi, beberapa metrik yang biasanya digunakan adalah Akurasi,Recall,heatmap dan F1-Score untuk melihat kesalahan prediksi dengan lebih detail. Pengembangan model akan menggunakan beberapa algoritma machine learning yaitu Random Forest dan Gradient Boosting. Dari kedua model ini, akan dipilih satu model yang memiliki nilai kesalahan prediksi terkecil. Dengan kata lain, kita akan membuat model seakurat mungkin, yaitu model dengan nilai kesalahan sekecil mungkin.

# Data Understanding

  Pada proyek ini menggunakan dataset "Stunting Toddler (Balita) Detection"
Dataset ini merupakan kumpulan data berdasarkan rumus z-score penentuan stunting menurut WHO (World Health Organization), yang berfokus pada deteksi stunting pada balita (bayi dibawah lima tahun) yang dapat digunakan untuk membantu peneliti dan praktisi di bidang kesehatan dan gizi anak dalam perencanaan strategi pencegahan dan pengobatan stunting pada balita yang lebih baik. Dataset diambil dari Kaggle : https://www.kaggle.com/datasets/rendiputra/stunting-balita-detection-121k-rows
# Variabel-variabel pada dataset Stunting Toddler (Balita) Detection adalah sebagai berikut:
1. Umur (Bulan): Mengindikasikan usia balita dalam bulan. Rentang usia ini penting untuk menentukan fase pertumbuhan anak dan membandingkannya dengan standar pertumbuhan yang sehat. (Umur 0 sampai 60 bulan)
2. Jenis Kelamin: Terdapat dua kategori dalam kolom ini, 'laki-laki' (male) dan 'perempuan' (female). Jenis kelamin merupakan faktor penting dalam analisis pola pertumbuhan dan risiko stunting.
3. Tinggi Badan: Dicatat dalam centimeter, tinggi badan adalah indikator utama untuk menilai pertumbuhan fisik balita. Data ini memungkinkan peneliti untuk menentukan apakah pertumbuhan anak sesuai dengan standar usianya.
4. Status Gizi: Kolom ini dikategorikan menjadi 4 status - 'severely stunting', 'stunting', 'normal', dan 'tinggi'. 'Severely stunting' menunjukkan kondisi sangat serius (<-3 SD), 'stunting' menunjukkan kondisi stunting (-3 SD sd <-2 SD), 'normal' mengindikasikan status gizi yang sehat (-2 SD sd +3 SD), dan 'tinggi' (height) menunjukkan pertumbuhan di atas rata-rata (>+3 SD). Kategori ini membantu dalam identifikasi cepat dan intervensi bagi anak-anak yang berisiko atau mengalami masalah pertumbuhan.
# Tahapan yang Dilakukan
1. Langkah pertama adalah mengimport seluruh library yang dibutuhkan dalam penelitian ini. Ada beberapa library yang digunakan seperti pandas,numpy,matplotlib,dan searborn.
2. Proses Load dataset proses ini dilakukan untuk melihat keseluruhan isi dataset yang kita gunakan. Dataset ini berupa format csv dan untuk membacanya kita perlu menuliskan kode 'pd.read_csv(url dataset disimpan)'.
Dataset ini terdiri dari 120999 rows dan 4 columns. Dimana 4 kolom tersebut terdiri dari kolom umur,jenis kelamin,tinggi badan,dan status gizi.
3. Proses Melihat Informasi Dataset
Untuk melihat informasi keseluruhan dari datasetnya kita menggunakan 'DataFrame.info()'. Kode 'info()' ini menampilkan informasi seperti range index,jumlah kolom dan namanya,type datanya,memori yang digunakan,dan lainnya. Disini terlihat nama kolom beserta type datanya. Untuk kolom jenis kelamin dan status gizi bertype data object,kolom umur type data integer,dan tinggi badan type data float,yang berarti untuk integer dan float ada 1 sedangkan untuk type data object ada 2. Untuk memori yang digunakannya sekitar 3.7+ MB dan RangeIndex: 120999 entries, 0 to 120998.
4. Mengecek deskripsi statistik data dengan fitur describe()
Fungsi describe() memberikan informasi statistik pada masing-masing kolom, antara lain:
- Count adalah jumlah sampel pada data.
- Mean adalah nilai rata-rata.
- Std adalah standar deviasi.
- Min yaitu nilai minimum setiap kolom.
- 25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas     interval dalam empat bagian sebaran yang sama.
- 50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
- 75% adalah kuartil ketiga.
- Max adalah nilai maksimum.
  Pada proses ini menggunakan kolom yang memiliki variabel numerik atau angka,karena disini yang bertype numerik ada dua yaitu kolom umur dan tinggi badan.
5. Menangani Missing Value
Missing value adalah data yang hilang atau kosong dalam sebuah dataset. Disini kita menggunakan kode 'isnull().sum()' untuk memeriksa missing value yang terdapat di dataset. Dan hasilnya menunjukkan bahwa dalam dataset "Stunting Toddler (Balita) Detection" tidak memiliki missing value atau bisa diartikan bahwa dataset ini sudah bersih dan bisa langsung digunakan untuk proses pemodelan.
6. Menangani Outlier dengan Fitur IQR Method
  IQR adalah singkatan dari Inter Quartile Range. Kuartil dari suatu populasi adalah tiga nilai yang membagi distribusi data menjadi empat sebaran. Seperempat dari data berada di bawah kuartil pertama (Q1), setengah dari data berada di bawah kuartil kedua (Q2), dan tiga perempat dari data berada di kuartil ketiga (Q3). Dengan demikian interquartile range atau IQR = Q3 - Q1. Untuk mengeceknya, kita akan menggunakan teknik visualisasi, yaitu jenis boxplot. Boxplot menunjukkan ukuran lokasi dan penyebaran, serta memberikan informasi tentang simetri dan outliers. Boxplot bisa digambarkan secara vertikal maupun horizontal.
  
Berdasakan visualisasi boxplot tersebut terlihat bahwa pada kolom umur dan tinggi badan tidak ada outlier,karena datasetnya sudah bersih dan tidak memiliki missing value.


  
![Screenshot 2025-02-16 233858](https://github.com/user-attachments/assets/71e5e8cf-0476-492c-9093-11065aa9ee75)


![Screenshot 2025-02-16 233846](https://github.com/user-attachments/assets/dc489580-60e9-4dab-97bb-ca3339fc6c9d)



7. Mengecek Ukuran Dataset
Untuk memeriksa ukuran dari dataset kita menggunakan 'shape()'. Hasil yang didapatkan adalah Dataset memiliki 120,999 baris dan 4 kolom.
8. Proses analisis data dengan teknik Univariate EDA
Proses pertama yang dilakukan dalam analisis data dengan teknik univariate adalah membagi data berdasarkan jenis variabelnya yaitu berdasarkan fitur numerik dan kategorikal. Fitur numerik terdiri dari kolom umur dan tinggi badan,sedangkan fitur kategori terdiri dari jenis kelamin dan status gizi.

a. Pendistribusian untuk mengetahui status gizi berdasarkan jenis kelamin

  Untuk melihat apakah ada pola tertentu antara status gizi dengan jenis kelamin,kita bisa melihatnya dengan visualisasi bar chart. Dimana dalam bart chart ini ada dua sumbu,sumbu x (horizontal) berisi status gizi yang terdiri dari empat (severely stunting, stunting, normal, tinggi). Sedangkan sumbu Y (vertical) menunjukkan jumlah dataset dalam setiap data di status gizi. Jumlah balita ini diperoleh karena sns.countplot() secara otomatis menghitung jumlah data dari dataset dan menampilkannya dalam bentuk grafik. Kemudian disini juga menggunakan 'Hue=' untuk perbandingan warna antara jenis kelamin laki-laki dan perempuan,untuk warna orange menandakan perempuan dan biru laki-laki.



![Screenshot 2025-02-16 234530](https://github.com/user-attachments/assets/07d61030-4eec-494e-bc63-23522b73d091)




Hasil visualisasi ini adalah diperoleh

- Status gizi yang paling besar adalah normal pada perempuan dibandingkan laki-laki,yang artinya mengindikasikan status gizi yang sehat (-2 SD sd +3 SD).
- Status gizi 'Severely stunting' lebih banyak dibandingkan 'stunting' dengan urutan kedua. Dan jenis kelamin dengan status gizi ini yang unggul sedikit adalah laki-laki dibanding perempuan. Hal ini harus menjadi perhatian para tenaga kesehatan karena 'Severely stunting' menunjukkan kondisi sangat serius (<-3 SD) dan perlu perhatian khusus.
- Status gizi 'Tinggi' perempuan lebih unggul sedikit dibandingkan laki-laki. Dan 'tinggi' (height) menunjukkan pertumbuhan di atas rata-rata (>+3 SD).
- Status gizi ' stunting' berada diposisi terakhir dengan jumlah antara perempuan dan laki-laki hampir setara. Kondisi ini menunjukkan kondisi stunting (-3 SD sd <-2 SD) dan harus menjadi perhatian khusus tenaga kesehatan dan penelitian juga.
- 
  Kesimpulannya adalah sebagian besar balita memiliki pertumbuhan yang baik karena berdasarkan grafik yang paling dominan adalah kondisi 'normal' walapun sebagian kondisi balita memprihatinkan karena ada pada kondisi 'several stunting'. Dan untuk distribusi laki-laki dan perempuan hampir mirip di setiap kategori, berarti tidak ada perbedaan signifikan antara jenis kelamin dalam status gizi.
  
b. Pendistribusian fitur kategorikal yang kedua di kolom status gizi



![Screenshot 2025-02-16 234544](https://github.com/user-attachments/assets/e894cd2a-0452-4921-b109-3db35116a4aa)



  Berdasarkan deskripsi variabel, urutan kategori status gizi ini dari yang paling tinggi ke yang paling rendah adalah normal,severely stunted ,tinggi,dan stunting. Dari grafik ini,dapat disimpulkan bahwa kondisi gizi anak-anak bayi dan balita masih dominan normal yang berarti memiliki pertumbuhan yang baik. Tetapi perlu diperhatikan juga bahwa di urutan kedua kondisinya severely stunted yang berarti kondisi ini sangat buruk dan harus segera diberikan perhatian khusus untuk ditangani. Karena tidak dapat kita pungkiri kondisi ini cukup tinggi dibeberapa negara atau kota yang kekurangan perhatian sehingga kondisi bayi dan balitanya sudah ditahap bahaya melibihi stunting. Kemudian untuk status gizi 'tinggi' ada diurutan ketiga yang mungkin hanya terjadi dibeberapa atau sedikit anak yang pertumbuhannya lebih cepat dibandingkan usianya. Dan terakhir kondisi stunting,walaupun diurutan terakhir tetap harus diperhatikan agar kondisi gizi anak masih bisa diperbaiki menjadi normal.
  
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

1. Encoding Fitur Kategori Untuk fitur encoding,salah satu teknik umum yang digunakan adalah label encoding. Karena Prediksi status gizi adalah tujuan yang ingin dicapai. Seperti yang kita ketahui,status gizi adalah variabel kategori karena status gizi punya tingkatan/urutan tetapi tidak berupa nilai. Jadi teknik label encoding sangat cocok untuk prediksi yang bersifat kategori. Library scikit-learn menyediakan fungsi ini untuk mendapatkan fitur baru yang sesuai sehingga dapat mewakili variabel kategori. Kita memiliki tiga variabel kategori dalam dataset kita, yaitu 'Jenis Kelamin, dan 'Status Gizi'. Dan sekarang output untuk kolom jenis kelamin menjadi 0 (untuk gender laki-lai) dan 1 (untuk gender perempuan). Sama Halnya dengan kolom jenis kelamin,kolom status gizi pun sudah berubah menjadi variabel numerik juga.

   
Kode: 


![Screenshot 2025-02-16 235248](https://github.com/user-attachments/assets/b2879a79-4e8b-4b0f-b2b7-c2f5b79021dc)



Output: 


![Screenshot 2025-02-16 235306](https://github.com/user-attachments/assets/2d476292-ea51-4669-b93d-9b635df4070e)




2. Pembagian dataset dengan fungsi train_test_split dari library sklearn. Membagi dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus kita lakukan sebelum membuat model. Tujuannya adalah agar kita tidak mengotori data uji dengan informasi yang kita dapat dari data latih. proporsi pembagian data latih dan uji adalah 80:20. Dan kita dapat memperoleh hasil dari proses train_test_split ini seperti kode dibawah.

   
Kode:


![Screenshot 2025-02-16 235515](https://github.com/user-attachments/assets/e63a2e97-2f3c-4974-997f-ffc35d9b7c66)



Output: 


![Screenshot 2025-02-16 235648](https://github.com/user-attachments/assets/cabe63a4-89bf-405f-bc35-b5cd618198ae)



3. Standarisasi Standardisasi adalah teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan. Untuk fitur numerik, kita tidak akan melakukan transformasi dengan one-hot-encoding seperti pada fitur kategori. Kita akan menggunakan teknik StandarScaler dari library Scikitlearn, StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi. StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1. Fitur standarisasi diterapkan pada data train dan data test dari kolom numerik 'umur' dan 'tinggi badan'. Dan sekarang outputnya telah menjadi nilai mean = 0 dan standar deviasi = 1.

   
kode: 


![Screenshot 2025-02-16 235613](https://github.com/user-attachments/assets/e34633f8-777a-41ba-b662-825d784382d8)

Output:


![Screenshot 2025-02-16 235632](https://github.com/user-attachments/assets/862e9ec1-c3d1-40f8-8197-501175ab41de)



# Modeling

  Pada tahap ini, menggunakan algoritma Random Forest untuk membangun model klasifikasi yang dapat memprediksi status gizi balita berdasarkan fitur-fitur seperti usia, jenis kelamin, dan tinggi badan. Berikut adalah penjelasan tahapan dan parameter yang digunakan:
# Persiapan Data:
1. Data dibagi menjadi fitur (X) dan target (y). Fitur meliputi Umur (bulan), Tinggi Badan (cm), dan Jenis Kelamin, sedangkan target adalah Status Gizi.
2. Data dibagi menjadi data latih (X_train, y_train) dan data uji (X_test, y_test) dengan proporsi 80:20 menggunakan train_test_split.
# Pelatihan Model:
1. Model Random Forest dilatih menggunakan data latih (X_train, y_train).
2. Parameter yang digunakan:
- n_estimators=50: Jumlah pohon keputusan dalam Random Forest.
- max_depth=16: Kedalaman maksimum setiap pohon keputusan.
- random_state=55: Untuk memastikan hasil yang konsisten.
- n_jobs=-1: Menggunakan semua core CPU untuk mempercepat pelatihan.
# Prediksi:
Model digunakan untuk memprediksi status gizi pada data uji (X_test), dan hasilnya disimpan di y_pred_RF.
# Kelebihan dan Kekurangan Random Forest:
- Kelebihan:
1. Dapat menangani data numerik dan kategorikal.
2. Tahan terhadap overfitting karena menggunakan banyak pohon keputusan.
3. Dapat mengukur importance fitur, yang membantu dalam memahami pola data.
- Kekurangan:
1. Membutuhkan waktu pelatihan yang lebih lama dibandingkan algoritma lain seperti Logistic Regression.
2. Kurang interpretatif dibandingkan model sederhana seperti Decision Tree.

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
