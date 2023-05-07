# -*- coding: utf-8 -*-
"""Predictive Analytics  Nurul Tazkiyah.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DxbuU9M10zG_UqYxz-bl2ERV1iepgJdJ

# **Predictive Analytics Ford Car Oleh Nurul Tazkiyah Adam**

# **1. Data Understanding**
- ### **Import Library**
"""

# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

"""- ### **Menyiapkan Dataset**

  Sumber dataset berasal dari Kaggle yang dapat dilihat melalui [tautan ini](https://www.kaggle.com/datasets/adhurimquku/ford-car-price-prediction). Selanjutnya, memasukkan dataset ke Google Drive pribadi sehingga dapat di download pada Google Colab dengan cara berikut ini

"""

# download dataset dari google drive https://drive.google.com/file/d/1t5jVU93rMo0geyVFbo2Dzm1V_ekX_e6A/view?usp=sharing
!gdown 1t5jVU93rMo0geyVFbo2Dzm1V_ekX_e6A

"""
- ### **Menampilkan Isi Dataset**"""

ford = pd.read_csv('ford.csv')
ford.head()

"""- ### **Exploratory Data Analysis**

Pada dataset Ford terdapat variabel:

- `model` ->  macam-macam merek pada mobil Ford.
- `year` -> tahun model mobil diproduksi.
- `price` -> harga mobil (satuan dollar / $).
- `transmission` -> transmission pada. mobil (Automatic, Manual, Semi-Auto)
- `mileage` -> jarak tempuh yang dapat dilalui mobil.
- `fuelType` -> tipe bahan bakar mobil (Petrol, Diesel, Hybrid, Electric, dll).
- `tax` -> pajak tahunan.
- `mpg` -> efesiensi bahan bakar.
- `engineSize` -> kapasistas mesin pada mobil.

dengan detail informasi jumlah dan tipe data berikut ini,
"""

ford.info()

"""terdapat **6 data numerik** yakni: `year`, `price`, `mileage`, `tax`, `mpg`, dan `engineSize` 
lalu **3 data kategoris** yakni: `model`, `transmission`, dan `fuelType`.


"""

ford.describe()

"""keterangan:

- **count** adalah jumlah sampel pada data.
- **mean** adalah nilai rata-rata.
- **std** adalah standar deviasi.
- **min** yaitu nilai minimum setiap kolom.
- **25%** adalah kuartil* pertama (Q1)
- **50%** adalah kuartil* kedua (Q2) atau  median (nilai tengah).
- **75%** adalah kuartil* ketiga (Q3).
- **Max** adalah nilai maksimum

** Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.*

- ### **Menangani Missing Values**
"""

ford.isnull().sum()

"""Seluruh data pada variable dataset terisi, akan tetapi perlu dicek kembali apakah ada data yang bernilai 0."""

mileage =(ford.mileage == 0).sum()
mpg =(ford.mpg == 0).sum()
engineSize =(ford.engineSize == 0).sum()

print('nilai 0 pada kolom mileage sebanyak', mileage)
print('nilai 0 pada kolom mpg sebanyak', mpg)
print('nilai 0 pada kolom engineSize sebanyak', engineSize)

"""Setelah dilakukan pengecekan terdapat 51 data yang bernilai 0 pada engineSize. Selanjutnya, mengecek jumlah nilai 0 terbanyak pada engineSize."""

ford.loc[ford['engineSize']==0]

"""Kemudian, menghapus baris data pada kolom `mileage`, `mpg`, `engineSize` yang bernilai 0.


"""

ford = ford.loc[(ford[['mileage','mpg','engineSize']]!=0).all(axis=1)]
ford.shape

"""Pengecekan data setelah menangani missing value."""

ford.describe()

"""- ### **Menangani Outliers**"""

plt.subplots(figsize=(10,7))
sns.boxplot(data=ford).set_title("Ford Car")
plt.show()

"""**Outlier** adalah sampel yang memiliki nilai yang sangat jauh dari cakupan umum data utama dan hasil pengamatan yang kemunculannya sangat jarang serta hasil pengamatannya berbeda dari data lainnya. Jika dilihat dari plot diatas, outliers terbanyak pada variable mileage dengan 175.000-an outliers.

kemudian, menangani outliers dengan persamaan:
- Batas bawah = `Q1 - 1.5 * IQR`
- Batas atas = `Q3 + 1.5 * IQR`
- Kemudian membuat rumus IQR (Inter Quartile Range) `IQR = Q3 - Q1`
"""

Q1 = ford.quantile(0.25)
Q3 = ford.quantile(0.75)
IQR = Q3-Q1
ford=ford[~((ford<(Q1-1.5*IQR))|(ford>(Q3+1.5*IQR))).any(axis=1)]

ford.shape

"""Didapatkan data sebanyak 16.450 sampel setelah menangani outliers."""

plt.subplots(figsize=(10,7))
sns.boxplot(data=ford).set_title("Ford Car")
plt.show()

"""sangat terlihat perubahan setelah pembersihan outliers. Salah satunya pada mileage, outliers menjadi 60.000-an yang sebelumnya 175.000-an.

- ### **Univariate Analysis**

Membagi fitur pada dataset menjadi dua bagian, yaitu numerical dan categorical.
"""

numerical = ['year','price','mileage','tax','mpg','engineSize']
categorical = ['model','transmission','fuelType']

"""- **Analisis fitur categorical: Model**"""

feature = categorical[0]
count = ford[feature].value_counts()
percent = 100*ford[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature)

"""- **Analisis fitur categorical: Transmission**"""

feature = categorical[1]
count = ford[feature].value_counts()
percent = 100*ford[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature)

"""- **Analisis fitur categorical: Fuel Type**"""

feature = categorical[2]
count = ford[feature].value_counts()
percent = 100*ford[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature)

"""- **Analisis Fitur Numerical: Year, Price, Mileage, Tax, Mpg dan EngineSize.**"""

ford.hist(bins=50, figsize=(20,15))
plt.show()

"""- ### **Multivariate Analysis**

Mengecek rata-rata price terhadap masing-masing fitur kategoris. Hal ini dilakukan untuk mengetahui pengaruh fitur kategoris terhadap price.
"""

categorical_features = ford.select_dtypes(include='object').columns.to_list()
for col in categorical_features:
    sns.catplot(x=col, 
                y='price', 
                kind='bar', 
                dodge=False, 
                height = 7, 
                aspect= 3, 
                data=ford, 
                palette='Set3')
    plt.title('Rata-rata "price" relatif terhadap - {}'.format(col))

"""Dengan mengamati rata-rata harga relatif terhadap fitur kategori di atas, kita memperoleh insight sebagai berikut:

- Pada `model`, rata-rata harga cenderung berbeda. Rentangnya berada antara 500 hingga 20.000-an. Grade tertinggi yaitu grade up memiliki harga rata-rata terendah diantara grade lainnya. Sehingga, fitur model memiliki pengaruh atau dampak yang kecil terhadap rata-rata harga.
- Pada `transmission`, rata-rata transmission yang paling rendah adalah transmission manual dan harganya pun rendah dibandingkan dengan transmission automatic dan semi-auto, hal ini menunjukkan bahwa fitur transmission memiliki pengaruh yang tinggi terhadap harga.
- Pada `fuelType`, pada umumnya, fueltype yang memiliki graden lebih tinggi memiliki harga yang tinggi juga, hal ini menunjukkan bahwa fitur fuelType memiiki pengaruh yang tinggi terhadap harga.

Kesimpulan akhir, fitur kategori memiliki pengaruh yang tinggi terhadap harga.

- **Kolerasi Fitur Numerik dengan Fitur Target menggunakan fungsi corr()**
"""

sns.pairplot(ford, diag_kind='kde')

"""- **Kolerasi Fitur Numerik Menggunakan Heatmap Correlation Matrix**"""

plt.figure(figsize=(10,8))
correlation_matrics = ford.corr().round(2)
sns.heatmap(data=correlation_matrics, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matrix Kolerasi Fitur Numerik', size=20)

"""Berdasarkan diagram heatmap di atas, disimpulkan bahwa:

- Rentang nilai dari 1 sampai -0.67.
- Jika nilai mendekati 1, maka korelasi antar fitur numerik semakin kuat positif.
- Jika nilai mendekati 0, maka korelasinya semakin rendah atau semakin tidak ada korelasi.
- Jika nilai mendekati -1, maka korelasi antar fitur numerik semakin kuat negatif.
- Korelasi antar fitur numerik yang memiliki korelasi positif dengan fitur `price` yakni fitur `tax`, `year`, dan `engineSize` (0.4 sampai 0.64)
- Sedangkan fitur `mileage` dan `mpg` memiliki korelasi yang sangat kecil (-0.36 sampai -0.48). Sehingga, fitur tersebut dapat di-drop.
"""

ford.drop(['mileage'], inplace=True, axis=1)
ford.drop(['mpg'], inplace=True, axis=1)
ford.head()

"""# **2. Data Preparation**
- ### **Encoding Fitur Kategoris**
"""

ford = pd.concat([ford, pd.get_dummies(ford['model'], prefix='model', drop_first=True)], axis=1)
ford = pd.concat([ford, pd.get_dummies(ford['transmission'], prefix='transmission', drop_first=True)], axis=1)
ford = pd.concat([ford, pd.get_dummies(ford['fuelType'], prefix='fuelType', drop_first=True)], axis=1)
ford.drop(['model','transmission','fuelType'], axis=1, inplace=True)
ford.head()

"""- ### **Reduksi Dimensi PCA**"""

sns.pairplot(ford[['engineSize','tax']], plot_kws={'s':2})

"""- **Aplikasi Class PCA**"""

pca = PCA(n_components=2, random_state=123)
pca.fit(ford[['engineSize','tax']])
princ_comp = pca.transform(ford[['engineSize','tax']])

"""- **Informasi Kedua Komponen**

"""

pca.explained_variance_ratio_.round(2)

"""- **Membuat Fitur dengan nama 'feature'**

fitur ini untuk mengganti fitur engineSize dan tax.
"""

pca = PCA(n_components=1, random_state=123)
pca.fit(ford[['engineSize','tax']])
ford['feature'] = pca.transform(ford.loc[:, ('engineSize','tax')]).flatten()
ford.drop(['engineSize','tax'], axis=1, inplace=True)

"""- ### **Membagi Data Latih dan Data Uji dengan Train Test Split**"""

x = ford.drop(['price'], axis=1)
y = ford['price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

"""- ### **Standarisasi**
Standarisasi pada fitur numerik yaitu year dan feature menggunakan StandardScaler untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma.
"""

numerical = ['year','feature']
scaler = StandardScaler()
scaler.fit(x_train[numerical])
x_train[numerical] = scaler.transform(x_train.loc[:, numerical])
x_train[numerical].head()

x_train[numerical].describe().round(4)

"""# **3. Model Development**
Menggabungkan tiga model algoritma yang akan digunakan yaitu K-Nearest Neighbor (KNN), Random Forest, dan Boosting Algorithm. Kemudian, mencari performa yang paling baik dari ketiga algoritma tersebut.

- **menyiapkan data frame**
"""

models = pd.DataFrame(index=['train_mse','test_mse'],
                    columns=['KNN', 'RandomForest', 'Boosting'])

"""- ### **K-Nearest Neighbor**"""

knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_train)

"""- ### **Random Forest**"""

RF = RandomForestRegressor(n_estimators=45, max_depth=16, random_state=55, n_jobs=-1)
RF.fit(x_train, y_train)

models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(x_train), y_true=y_train)

"""### **Boosting Algorithm**"""

boosting = AdaBoostRegressor(n_estimators=50, learning_rate=0.05, random_state=55)
boosting.fit(x_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(x_train), y_true=y_train)

"""# **4. Evaluasi Model**
- ### **Mengukur seberapa kecil nilai error MSE**
"""

x_test.loc[:, numerical] = scaler.transform(x_test[numerical])

mse = pd.DataFrame(columns=['train','test'],index=['KNN','RF','Boosting'])
model_dict = {'KNN':knn, 'RF':RF, 'Boosting': boosting}
for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(x_train))/1e3
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(x_test))/1e3
mse

fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

"""Dari gambar di atas, terlihat bahwa, model RF memberikan nilai eror yang paling kecil. Model inilah yang dapat digunakan sebagai model terbaik untuk melakukan prediksi harga mobil Ford.

- ### **Nilai Akurasi Model**
"""

knn_accuracy = knn.score(x_test, y_test)*100
rf_accuracy = RF.score(x_test, y_test)*100
boosting_accuracy = boosting.score(x_test, y_test)*100

list_evaluasi = [[knn_accuracy],
            [rf_accuracy],
            [boosting_accuracy]]
evaluasi = pd.DataFrame(list_evaluasi,
                        columns=['Accuracy (%)'],
                        index=['K-Nearest Neighbor', 'Random Forest', 'Boosting'])
evaluasi

"""Dari hasil evaluasi di atas dapat memberikan informasi bahwa model Algorithma Random Forest mencapai akurasi hingga 90% lebih, model KNN 89% lebih, sedangkan model Boosting masih termasuk rendah yakni 60%

### **Prediksi**
"""

prediksi = x_test.iloc[:1].copy()
pred_dict = {'y_true':y_test[:1]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)

pd.DataFrame(pred_dict)

"""dapat dilihat dari prediksi di atas bahwa prediki dengan model RF memberikan hasil yang paling mendekati nilai y_true dibanding dengan kedua model lainnya.

Referensi:


*   [Prediksi Harga Bitcoin](https://github.com/AzharRizky/Bitcoin-Predictive-Anlaytics/blob/main/Submission_1_MLT.ipynb)
*   [Prediksi Harga Gemstone](https://github.com/chelizaaa/gemstone-predictive-analytics/blob/main/Gemstone%20Predictive%20Analytics.ipynb)
*   [Prediksi Harga Mobil](https://github.com/onedayxzn/prediksi-harga-mobil-volkswagen/blob/master/Submissiomv1_3.ipynb)
"""