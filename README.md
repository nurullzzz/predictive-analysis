# Laporan Proyek Machine Learning - Nurul Tazkiyah Adam

## Daftar Isi

- [Domain Proyek](#domain-proyek)
- [Business Understanding](#business-understanding)
- [Data Understanding](#data-understanding)
- [Data Preparation](#data-preparation)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Referensi](#referensi)

## Domain Proyek

**Ford Motor Company** atau **Ford** adalah salah satu perusahaan otomotif terbesar di dunia
yang berbasis dari Dearborn, Michigan, dan Amerika Serikat. Telah memproduksi dan
mendistribusikan hingga melintasi 6 benua. Karyawan sebanyak 229.000 dan pabrik sebanyak 90 secara worldwide, perusahaan Ford berafiliasi dengan perusahan otomotif lainnya seperti Lincoln, Mercury, Volvo dan Mazda. 

Sebagai perusahaan otomotif yang mendunia, Ford bukanlah pendatang baru di Indonesia. sejak tahun 1989, Ford telah hadir di Indonesia dan saat itu Ford di Indonesia diwakilkan oleh Indonesia Republic Motor Company (IRMC). Kemudian, pada 12 Juli 2000 PT Ford Motor Indonesia diresmikan sebagai Agen Tunggal Pemegang Merek (ATPM) Ford di Indonesia dan mulai mendistribusikan produk Ford sejak tahun 2001.

![Ford Car](https://cdn1-production-images-kly.akamaized.net/yllfpSJRJWuxYDrvy3hvPpF7lkQ=/1280x720/smart/filters:quality(75):strip_icc():format(webp)/kly-media-production/medias/1421007/original/002914500_1480476610-2016-fiesta-st-line-e1480457340628.jpg)  
**Gambar Mobil Ford**

Pada riset sebelumnya oleh Wina Novriyanti Lumban Toruan, Mahasiswa Studi Periklanan Universitas Indonesia, dengan judul Perencanaan Program Komunikasi Pemasaran Terpadu Ford Fiesta 2012-2013, menemukan rincian masalah bahwa meskipun perusahaan ford telah memiliki banyak penghargaan dunia, masih banyak konsumen yang tidak sadar akan keberadaan setiap keluaran model terbaru yang dikeluarkan oleh perusahaan Ford. Hal ini dikarenakan *awereness brand* perusahaan belum tercipta dengan baik. Untuk itu penelitian tersebut dirancang untuk menemukan konsep dan pesan yang lebih efektif dari iklan Ford Fiesta serta menindak lanjuti evaluasi dari iklan yang telah dirancang dalam menenutukan periklanan di tahun selanjutnya.

 Namun persaingan secara global dan mendunia, harga mobil Ford tidak hanya ditentukan dari harga permintaan pasar. Sebagai otomotif yang memiliki fitur sangat lengkap di kelasnya, Ford harus mampu memprediksi harga mobil berdasarkan kualitas produknya sendiri dengan baik juga. 

Hasil riset terkait dapat dilihat dari tautan berikut

- [Perencanaan Program Komunikasi Pemasaran Terpadu Ford Fiesta 2012-2013](http://lib.ui.ac.id/file?file=digital/20312843-S43587-Perencanaan%20program.pdf)

## Business Understanding

### Problem Statements

1. Apa saja tahap persiapan data yang perlu dilakukan sebelum digunakan untuk pelatihan model *machine learning*?
2. Bagaimana cara menentukan model *machine learning* terbaik untuk memprediksi harga dari Mobil Ford?

### Goals

1. Melakukan tahap persiapan data atau *data preparation*, agar data yang digunakan dapat dipakai untuk melatih model *machine learning*.
2. Menentukan model *machine learning* yang memiliki tingkat *error* model algoritma *machine learning* paling rendah dan akurasi paling tinggi untuk memprediksi harga dari mobil Ford.

### Solution Statements

Berdasarkan rumusan masalah dan tujuan di atas, maka disimpulkan beberapa solusi yang dapat dilakukan untuk mencapai tujuan dari proyek ini, yaitu:
 1. Tahap persiapan data dilakukan dengan menggunakan beberapa teknik persiapan data, yaitu:
    - Mengatasi data pencilan (outliers) dengan Metode IQR.
    - Melakukan proses *encoding* fitur kategori dataset, yaitu kategori `model`, `transmission`, `fuelType`. sehingga diperoleh fitur baru yang mewakili masing-masing variabel kategori.
    - Melakukan proses reduksi feature numerical menggunakan *Principal Component Analysis* (PCA) untuk mengurangi jumlah fitur numerik dengan tetap mempertahankan informasi pada data, sehingga diperoleh sebuah fitur baru yang merupakan hasil dari beberapa fitur numerik.
    - Melakukan proses membagi dataset menjadi data latih dan data uji dengan perbandingan 80 : 20 dari total seluruh dataset yang akan digunakan saat membuat model *machine learning*.
    - Melakukan proses standarisasi fitur numerik menjadi bentuk data yang lebih mudah dipahami dan diolah oleh model *machine learning*.

2. Tahap membuat model *machine learning* untuk memprediksi harga mobil Ford dilakukan menggunakan 3 model algoritma *machine learning*  yang berbeda dan kemudian akan dilakukan evaluasi model untuk membandingkan performa model yang terbaik. Algoritma yang akan digunakan, yaitu: 
   - Algoritma K-Nearest Neighbor
   - Algoritma Random Forest
   - Boosting Algorithm
   
## Data Understanding

Dataset yang digunakan adalah dataset [Ford Car Price Prediction](https://www.kaggle.com/datasets/adhurimquku/ford-car-price-prediction) yang diambil dari platform Kaggle. *File* yang digunakan berupa *file* csv, yaitu `ford.csv`.

Kemudian dilakukan proses *Exploratory Data Analysis* (EDA) yang merupakan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data.
  
  - Terdapat  17.966 baris (_records_ atau jumlah pengamatan) yang berisi informasi mengenai data riwayat harga **Mobil Ford**.
  - Pada dataset Ford terdapat 9 kolom dengan variabel:
    - `model` ->  macam-macam merek pada mobil Ford.
    - `year` -> tahun model mobil diproduksi.
    - `price` -> harga mobil (satuan dollar / $).
    - `transmission` -> transmission pada. mobil (Automatic, Manual, Semi-Auto)
    - `mileage` -> jarak tempuh yang dapat dilalui mobil.
    - `fuelType` -> tipe bahan bakar mobil (Petrol, Diesel, Hybrid, Electric, dll).
    - `tax` -> pajak tahunan.
    - `mpg` -> efesiensi bahan bakar.
    - `engineSize` -> kapasistas mesin pada mobil.
  - Dari 9 kolom di atas, terdapat **6 data numerik** yakni: `year`, `price`, `mileage`, `tax` dengan tipe data *int64* dan `mpg`, `engineSize` dengan tipe data *float64*. Lalu, **3 data kategoris** yakni: `model`, `transmission`, dan `fuelType` dengan tipe data *object*.
  - Deskripsi statistik
  **Hasil Tabel Deskripsi statistik**

 |       | year    | price  | mileage  | tax | mpg | engineSize |
   |-------|--------------|--------------|--------------|--------------|--------------|--------------|
   | count | 17966.000000 | 17966.000000 | 17966.000000 | 17966.000000 | 17966.000000 | 17966.000000 |
   | mean  | 2016.866470 | 12279.534844 | 23362.608761 | 113.329456 | 57.906980    | 1.350807  |
   | std   | 2.050336 | 4741.343657 | 19472.054349 | 62.012456 | 10.125696 | 0.432367 | 
   | min   | 1996.000000 | 495.000000 | 1.000000 | 0.000000	 |20.800000 | 0.000000 |
   | 25%   | 2016.000000 | 8999.000000 | 9987.000000 | 30.000000 | 52.300000 | 1.000000  |
   | 50%   | 2017.000000 | 11291.000000	 | 18242.500000	 | 145.000000 | 58.900000    | 1.200000| 
   | 75%   | 2018.000000 | 15299.000000 | 31060.000000 | 145.000000 | 65.700000    | 1.500000 |
   | max   | 2060.000000 | 54995.000000 | 177644.000000 | 580.000000 | 201.800000 | 5.000000 |

   keterangan:
    - **count** adalah jumlah sampel pada data.
    - **mean** adalah nilai rata-rata.
    - **std** adalah standar deviasi.
    - **min** yaitu nilai minimum setiap kolom.
    - **25%** adalah kuartil* pertama (Q1)
    - **50%** adalah kuartil* kedua (Q2) atau  median (nilai tengah).
    - **75%** adalah kuartil* ketiga (Q3).
    - **Max** adalah nilai maksimum
     **Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.*
- **Menangani Missing Value**
    Tidak terdapat baris yang tidak terisi pada variable dataset. Akan tetapi, menghapus data yang bernilai 0 pada mileage, mpg, dan engineSize. Pada engineSize terdapat 51 data bernilai 0. Sehingga, perlu dihapus pada baris tersebut agar pada fitur numerik nilai min(minimal) sudah tidak bernilai 0 lagi. hasil deskripsi data setelah menangani missing value adalah berikut ini.

 |       | year    | price  | mileage  | tax | mpg | engineSize |
   |-------|--------------|--------------|--------------|--------------|--------------|--------------|
   | count | 17915.000000| 17915.000000| 17915.000000 | 17915.000000 | 17915.000000 | 17915.000000 |
   | mean  | 2016.865197 | 12280.966118 | 23373.346414 | 113.329456 | 113.342004 | 1.354653 |
   | std   | 2.050336 | 4741.343657 | 19472.054349 | 62.012456 | 10.125696 | 0.426924| 
   | min   | 1996.000000 | 495.000000 | 1.000000 | 0.000000	 |20.800000 | 1.000000  |
   | 25%   | 2016.000000 | 8999.000000 | 9987.000000 | 30.000000 | 52.300000 | 1.000000  |
   | 50%   | 2017.000000 | 11291.000000	 | 18250.000000	 | 145.000000 | 58.900000 | 1.200000| 
   | 75%   | 2018.000000 | 15299.000000 | 31083.000000 | 145.000000 | 65.700000    | 1.500000 |
   | max   | 2060.000000 | 54995.000000 | 177644.000000 | 580.000000 | 201.800000 | 5.000000 |
 
- **Menangani Outliers**
   
   ![outliers](https://raw.githubusercontent.com/nurullzzz/predictive-analysis/main/outlier.png)   
   Terlihat jika di atas banyak terdapat outlier pada setiap variabel, lalu untuk mengatasinya nantinya penulis akan menerapkan batas bawah dan batas atas menggunakan metode IQR. 
- **Univariate Analysis**

   Proses univariate data analysis pada masing-masing fitur kategorial dan numerik.
   
   - Categorical Features
     
     ![univariate-categorical-model](https://raw.githubusercontent.com/nurullzzz/predictive-analysis/main/model.png)  
Pada fitur model, terdapat 19 kategori,  dengan persentase tertinggi terdapat pada kategori Ideal sebesar 37.7%.
     
     ![univariate-categorical-transmission](https://raw.githubusercontent.com/nurullzzz/predictive-analysis/main/transmission.png)
Pada fitur transmission, terdapat 3 kategori,  dengan persentase tertinggi terdapat pada kategori manual sebesar 87.0%.

        ![univariate-categorical-fueltype](https://raw.githubusercontent.com/nurullzzz/predictive-analysis/main/fueltype.png)

        Pada fitur fueltype, terdapat 5 kategori, yaitu petrol, diesel, hybrid, electric, dan other dengan persentase tertinggi pada terdapat pada kategori petrol sebesar 69.8%.
     
   - Numerical Features
     
     ![univariate-numerical](https://raw.githubusercontent.com/nurullzzz/predictive-analysis/main/numerical.png)
     Terlihat pada grafik bahwa data distribusi ada yang distribusi nilainya miring ke kanan (right-skewed) dan ada yang rata (zero-skewed). Hal ini akan berimplikasi pada model nantinya.
   
- **Multivariate Analysis**
    Proses multivariate data analysis pada masing-masing fitur kategorial dan numerik.
   - Categorical Features
     Melakukan pengecekan rata-rata harga terhadap masing-masing fitur kategori, yaitu model, transmission, dan fueltype untuk mengetahui pengaruh fitur tersebut terhadap harga.
     ![multivariate-categorical-model](https://raw.githubusercontent.com/nurullzzz/predictive-analysis/main/multi.png)
     ![multivariate-categorical-trans](https://raw.githubusercontent.com/nurullzzz/predictive-analysis/main/transmission%20multi.png)
     ![multivariate-categorical-fuel](https://raw.githubusercontent.com/nurullzzz/predictive-analysis/main/multi%20fuel.png)
    Berdasarkan histogram di atas, dapat disimpulkan:
    Dengan mengamati rata-rata harga relatif terhadap fitur kategori di atas, kita memperoleh insight sebagai berikut:
        - Pada `model`, rata-rata harga cenderung berbeda. Rentangnya berada antara 500 hingga 20.000-an. Grade tertinggi yaitu grade up memiliki harga rata-rata terendah diantara grade lainnya. Sehingga, fitur model memiliki pengaruh atau dampak yang kecil terhadap rata-rata harga.
        - Pada `transmission`, rata-rata transmission yang paling rendah adalah transmission manual dan harganya pun rendah dibandingkan dengan transmission automatic dan semi-auto, hal ini menunjukkan bahwa fitur transmission memiliki pengaruh yang tinggi terhadap harga.
        - Pada `fuelType`, pada umumnya, fueltype yang memiliki graden lebih tinggi memiliki harga yang tinggi juga, hal ini menunjukkan bahwa fitur fuelType memiiki pengaruh yang tinggi terhadap harga.
     
   - Numerical Features
     Adapun untuk melakukan pengecekan rata-rata harga terhadap masing-masing fitur numerik, untuk mengetahui pengaruh fitur tersebut terhadap harga.
     ![multivariate-numerical](https://raw.githubusercontent.com/nurullzzz/predictive-analysis/main/corrr.png)  
-  **Correlation Matrix**
   Pengecekan korelasi atau hubungan antar fitur numerik menggunakan *heatmap correlation matrix*.
   ![correlation-matrix](https://raw.githubusercontent.com/nurullzzz/predictive-analysis/main/metrix.png)  
   Berdasarkan diagram *heatmap* di atas, disimpulkan bahwa:
   - Rentang nilai dari 1 sampai -0.67.
   - -Jika nilai mendekati 1, maka korelasi antar fitur numerik semakin kuat positif.
   - Jika nilai mendekati 0, maka korelasinya semakin rendah atau semakin tidak ada korelasi.
   - Jika nilai mendekati -1, maka korelasi antar fitur numerik semakin kuat negatif.
   - Korelasi antar fitur numerik yang memiliki korelasi positif dengan fitur `price` yakni fitur `tax`, `year`, dan `engineSize` (0.4 sampai 0.64)
   - Sedangkan fitur `mileage` dan `mpg` memiliki korelasi yang sangat kecil (-0.36 sampai -0.48). Sehingga, fitur tersebut dapat di-drop.
   

## Data Preparation

Beberapa proses yang dilakukan yakni, *encoding* pada fitur kategori, reduksi dimensi dengan menggunakan Principal Component Analysis (PCA), pembagian data latih dan data uji, dan standarisasi data.

- **Encoding Fitur Kategori**

   Proses *encoding* fitur kategori yaitu model, transmission, dan fuelType dengan teknik *one-hot-encoding*, sehingga diperoleh fitur baru yang mewakili masing-masing variabel kategori.
   
- **Reduksi dimensi dengan PCA**

   teknik yang digunakan untuk mengurangi jumlah fitur dengan tetap mempertahankan informasi pada data dalam kasus ini adalah Principal Component Analysis (PCA) dengan tujuan mereduksi dimensi, mengekstraksi fitur, dan mentransformasi data dari "n-dimensional space" ke dalam sistem berkoordinat baru dengan dimensi m, di mana m lebih kecil dari n.
   
   ![reduksi-dimensi](https://raw.githubusercontent.com/nurullzzz/predictive-analysis/main/reduksi.png)  

   Hasil proporsi informasi dari fitur engineSize dan Tax dengan menggunakan Principal Component Analysis (PCA), yaitu
   `array([1., 0.])`
   
- **Pembagian Dataset**

   Perbandingan data latih dan data uji pada dataset, yakni 80 : 20. Jadi, total sampel data latih sebanyak 13.160 data dan total sampel data uji sebanyak 3.290 data.
   
- **Standarisasi**

   Proses standarisasi fitur numerik, yaitu `year` dan `feature` menggunakan StandardScaler sehingga fitur data menjadi bentuk yang lebih mudah diolah oleh model machine learning.

## Modeling 

Sebelum melakukan pengembangan model, dilakukan persiapan *dataframe* untuk menganalisis model dengan menggunakan algoritma `K-Nearest Neighbor (KNN), Random Forest, dan Boosting Algorithm.`
 
- **Algoritma K-Nearest Neighbor (KNN)**

   Algoritma KNN menggunakan kesamaan fitur untuk memprediksi nilai dari setiap data yang baru. KNN bekerja dengan cara membandingkan jarak satu sampel ke sampel pelatihan lain dan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif). Pada algoritma K-Nearest Neighbor menggunakan parameter `n-neighbors` dengan nilai k = 10.
   
   ```python
   knn = KNeighborsRegressor(n_neighbors=10)
   ```
    -   Kelebihan:
       Algoritma KNN merupakan algoritma yang sederhana dan mudah untuk diimplementasikan misalnya pada beberapa kasus klasifikasi, regresi dan pencarian.
  -   Kekurangan:
       Algoritma KNN menjadi lebih lambat secara signifikan seiring meningkatnya jumlah sampel dan/atau variabel independen.
   
- **Algoritma Random Forest**

   Algoritma Random Forest merupakan salah satu algoritma *supervised learning* yang digunakan untuk menyelesaikan masalah klasifikasi dan regresi. Random forest merupakan salah satu model machine learning yang termasuk ke dalam kategori *ensemble* (*group*) *learning*. Pada algoritma Random Forest menggunakan parameter `n-estimator` dengan jumlah 45 trees (pohon), `max-depth` dengan nilai kedalaman atau panjang pohon sebesar 16, `random-state` dengan nilai 55, dan `n-jobs` yang bernilai -1 yang berarti pekerjaan dilakukan secara paralel.
   
   ```python
   RF = RandomForestRegressor(n_estimators=45, max_depth=16, random_state=55, n_jobs=-1)
   ```
    -   Kelebihan :
    Algoritma Random Forest merupakan algoritma dengan pembelajaran paling akurat yang tersedia. Untuk banyak kumpulan data, algoritma ini menghasilkan pengklasifikasi yang sangat akurat, berjalan secara efisien pada data besar, dapat menangani ribuan variabel input tanpa penghapusan variabel, memberikan perkiraan variabel apa yang penting dalam klasifikasi, dan memiliki metode yang efektif untuk memperkirakan data yang hilang dan menjaga akurasi ketika sebagian besar data hilang.
  -   Kekurangan :
    Algoritma Random Forest overfiting untuk beberapa kumpulan data dengan tugas klasifikasi/regresi yang _bising/noise_. Selain itu, untuk data yang menyertakan variabel kategorik dengan jumlah level yang berbeda, Random Forest menjadi bias dalam mendukung atribut dengan level yang lebih banyak. Oleh karena itu, skor kepentingan variabel dari Random Forest tidak dapat diandalkan untuk jenis data ini.
   
- **Boosting Algorithm**

   Boosting Algoritma ini bertujuan untuk meningkatkan performa atau akurasi prediksi dengan menggabungkan beberapa model sederhana dan dianggap lemah (*weak learners*) sehingga membentuk suatu model yang kuat (*strong ensemble learner*). Pada algoritma Boosting menggunakan `n_estimators` sebanyak 50, parameter `learning-rate` dengan nilai bobot setiap *regressor* adalah 0.05, dan `random-state` dengan nilai 55.
   
   ```python
   boosting = AdaBoostRegressor(n_estimators=50, learning_rate=0.05, random_state=55)
   ```
    -   Kelebihan :
    Algoritma Boosting dapat mengurangi bias pada data. Selain itu prosedur Boosting cukup sederhana. Algoritma ini juga sangat powerful dalam meningkatkan akurasi prediksi dan Algoritma boosting sering mengungguli model yang lebih sederhana seperti logistic regression dan random forest.
     -   Kekurangan :
    AdaBoost sangat dipengaruhi oleh outlier.

Dari ketiga model *machine learning* dengan algoritma K-Nearest Neighbor, Random Forest, dan Boosting Algorithm, akan dilakukan pengujian performa dan memilih satu model dengan nilai error yang paling rendah, akurasi yang paling tinggi dan hasil prediksi yang paling mendekati.

## Evaluation

Pada tahap evaluasi, dilakukan pengujian model dengan ketiga algoritma yang telah dibuat pada tahap modeling. Sebelum melakukan evaluasi, dilakukan proses *scaling* pada fitur-fitur numerik pada data uji sehingga skala antara data latih dan data uji sama.

```python
x_test.loc[:, numerical] = scaler.transform(x_test[numerical])
```

Metrik evaluasi *Mean Squared Error* (MSE) digunakan untuk mengevaluasi besaran *error* atau kesalahan dalam model algoritma machine learning yang digunakan. MSE sendiri merupakan Mean Squared Error yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. MSE didefinisikan dalam persamaan berikut:

   ![mse](https://raw.githubusercontent.com/nurullzzz/predictive-analysis/main/mse.png)  

keterangan:
- n = jumlah dataset  
- sum = *summation*/penjumlahan  
- Yi = nilai sebenarnya  
- Yi^ = nilai prediksi  

Hasil perbandingan MSE ketiga algoritma yang digunakan yakni

|          | train      | test       |
|----------|------------|------------|
| KNN      | 1662.436891 | 1605.018533 |
| RF       | 1460.487585 | 1523.150922 |
| Boosting | 6457.216399 | 6183.160884 |

![evaluation-graph](https://raw.githubusercontent.com/nurullzzz/predictive-analysis/main/grafikmse.png) 

Dari gambar di atas, terlihat bahwa, model RF memberikan nilai eror yang paling kecil. Model inilah yang dapat digunakan sebagai model terbaik untuk melakukan prediksi harga mobil Ford. 

Selanjutnya menghitung nilai akurasi model dan didapatkan

|          | Accuracy(%)|
|----------|------------|
|K-Nearest Neighbor| 89.689553 | 
| Random Forest | 90.215461 |
| Boosting | 60.280116 |

Dari hasil evaluasi di atas dapat memberikan informasi bahwa model Algorithma Random Forest mencapai akurasi hingga 90% lebih, model KNN 89% lebih, sedangkan model Boosting masih termasuk rendah yakni 60%

Terakhir, pengujian prediksi model menggunakan data uji.

|      | y_true | prediksi_KNN | prediksi_RF | prediksi_Boosting |
|------|--------|--------------|-------------|-------------------|
| 12643 |10499   | 10441.6	   | 10442.6   | 10591.5       |

dapat dilihat dari prediksi di atas bahwa prediki dengan model RF memberikan hasil yang paling mendekati nilai y_true dibanding dengan kedua model lainnya.

Kesimpulan yang diperoleh dari hasil analisis dan pemodelan machine learning untuk kasus ini: model algoritma Random Forest adalah model yang cocok untuk memprediksi harga mobil Ford karena menghasilkan tingkat error yang paling rendah, memiliki akurasi yang paling tinggi, dan memberikan hasil prediksi yang paling mendekati dengan data sebenarnya jika dibandingkan dengan algoritma lainnya.

## Referensi

[1]  http://lib.ui.ac.id/file?file=digital/20312843-S43587-Perencanaan%20program.pdf

[2]  https://ilmudatapy.com/algoritma-k-nearest-neighbor-knn-untuk-klasifikasi.

[3]  https://www.trivusi.web.id/2022/08/algoritma-random-forest.html.

[4]  https://www.dqlab.id/algoritma-machine-learning-yang-perlu-dipelajari.

[5]  https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html


