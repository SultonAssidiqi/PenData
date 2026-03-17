# Data Understanding

Dataset yang digunakan adalah **Iris Flower Dataset**, yang memuat data pengukuran morfologi bunga iris. Dataset ini sering dimanfaatkan dalam penelitian klasifikasi karena strukturnya sederhana, kualitas datanya baik, dan pembagian kelasnya jelas.

### 1. Referensi Data

Data diperoleh dari platform Kaggle dengan nama *Iris Flower Dataset* (https://www.kaggle.com/datasets/arshid/iris-flower-dataset?resource=download). Dataset ini termasuk dataset klasik dalam bidang data mining dan machine learning yang berisi data pengukuran morfologi bunga iris dari tiga spesies, yaitu Setosa, Versicolor, dan Virginica. Dataset tersebut digunakan sebagai studi kasus untuk penerapan metode klasifikasi dalam proses penambangan data.

### 2. Deskripsi Dataset

Dataset terdiri dari 150 baris data, di mana setiap baris merepresentasikan satu sampel bunga iris. Dataset ini memiliki 5 atribut, yang terdiri dari 4 atribut fitur dan 1 atribut target. Data tersebut digunakan untuk mengklasifikasikan bunga iris ke dalam tiga spesies berdasarkan ukuran bagian-bagian bunganya.

### 3. Eksplorasi Dataset

Dalam proses mengidentifikasi dataset ini, python membantu untuk mempermudah pengidentifikasiannya.

##### - Upload file CSV

```
from google.colab import files
files.upload()
```

Proses upload file CSV dilakukan untuk memasukkan dataset ke dalam lingkungan Google Colaboratory agar dapat diproses menggunakan bahasa pemrograman Python. Dataset diunggah secara manual dari perangkat pengguna menggunakan fungsi files.upload() yang disediakan oleh Google Colab.

##### - Struktur Dataset

```
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("IRIS.csv")
df.head()
```

Berikut tampilan dari code di atas setelah di jalankan:


| index | sepal\_length | sepal\_width | petal\_length | petal\_width | species |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 0 | 5\.1 | 3\.5 | 1\.4 | 0\.2 | Iris-setosa |
| 1 | 4\.9 | 3\.0 | 1\.4 | 0\.2 | Iris-setosa |
| 2 | 4\.7 | 3\.2 | 1\.3 | 0\.2 | Iris-setosa |
| 3 | 4\.6 | 3\.1 | 1\.5 | 0\.2 | Iris-setosa |
| 4 | 5\.0 | 3\.6 | 1\.4 | 0\.2 | Iris-setosa |

Dataset memiliki 5 kolom, yaitu:

1. Sepal Length (cm) (Numeric)
Menunjukkan panjang kelopak luar bunga iris dalam satuan sentimeter.
2. Sepal Width (cm) (Numeric)
Menunjukkan lebar kelopak luar bunga iris dalam satuan sentimeter.
3. Petal Length (cm) (Numeric)
Menunjukkan panjang mahkota bunga iris dalam satuan sentimeter.
4. Petal Width (cm) (Numeric)
Menunjukkan lebar mahkota bunga iris dalam satuan sentimeter.
5. Species (Kategorikal - String)
Menunjukkan jenis bunga iris, yang terdiri dari tiga kategori:
    - Setosa
    - Versicolor
    - Virginica

##### -Statistik Deskriptif Awal

```
df.describe()
```

Statistik deskriptif digunakan untuk mengetahui gambaran umum data seperti nilai rata-rata, nilai minimum, dan maksimum dari setiap atribut numerik.

Dataset Iris memiliki 4 atribut numerik, yaitu:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Berikut contoh statistik deskriptifnya:


| index | sepal\_length | sepal\_width | petal\_length | petal\_width |
| :-- | :-- | :-- | :-- | :-- |
| count | 150\.0 | 150\.0 | 150\.0 | 150\.0 |
| mean | 5\.843333333333334 | 3\.0540000000000003 | 3\.758666666666666 | 1\.1986666666666668 |
| std | 0\.8280661279778629 | 0\.4335943113621737 | 1\.7644204199522617 | 0\.7631607417008414 |
| min | 4\.3 | 2\.0 | 1\.0 | 0\.1 |
| 25% | 5\.1 | 2\.8 | 1\.6 | 0\.3 |
| 50% | 5\.8 | 3\.0 | 4\.35 | 1\.3 |
| 75% | 6\.4 | 3\.3 | 5\.1 | 1\.8 |
| max | 7\.9 | 4\.4 | 6\.9 | 2\.5 |

Berdasarkan hasil analisis statistik deskriptif, dataset Iris terdiri dari 150 data pada setiap atribut numerik, sehingga dapat disimpulkan tidak terdapat missing value. Nilai rata-rata (mean) menunjukkan bahwa panjang sepal bunga iris adalah 5.84 cm, lebar sepal 3.05 cm, panjang petal 3.76 cm, dan lebar petal 1.20 cm. Standar deviasi tertinggi terdapat pada atribut petal length sebesar 1.76, yang menunjukkan bahwa variasi panjang petal lebih besar dibandingkan atribut lainnya. Nilai minimum dan maksimum menunjukkan rentang data yang masih dalam batas wajar, yaitu sepal length antara 4.3 hingga 7.9 cm dan petal width antara 0.1 hingga 2.5 cm. Selain itu, nilai kuartil (25%, 50%, dan 75%) menunjukkan distribusi data yang relatif stabil tanpa adanya penyimpangan ekstrem yang signifikan. Secara umum, dataset memiliki distribusi yang baik dan layak digunakan untuk proses analisis dan pemodelan pada tahap selanjutnya.

##### - Pengecekan Data Duplikat

```
df.duplicated().sum()
```

Code di atas merupakan code untuk pengecekan data duplikat, berikut untuk hasil dari pengecekan data duplikat:

```
np.int64(3)
```

Jadi setelah dilakukan pengecekan, terdapat data duplikat yaitu ada 3 data duplikat.

##### - Pengecekan Data Null

```
df.isnull().sum()
```

Berikut untuk tampilan hasil dari pengecekan data Null:


|  |  |
| :-- | :-- |
| sepal_length | 0 |
| sepal_width | 0 |
| petal_length | 0 |
| petal_width | 0 |
| species | 0 |

### 4. Verifikasi Data

Berdasarkan hasil verifikasi, dataset Iris memiliki 150 data dengan 4 atribut numerik dan 1 atribut kategorikal (species). Hasil pemeriksaan menunjukkan bahwa seluruh atribut memiliki jumlah data yang sama (count = 150), sehingga dapat disimpulkan tidak terdapat missing value pada dataset. Tipe data juga sudah sesuai, di mana sepal length, sepal width, petal length, dan petal width bertipe numerik, sedangkan species bertipe kategorikal.

Selain itu, hasil pengecekan menunjukkan adanya 3 data duplikat dalam dataset. Keberadaan duplikasi ini bukan merupakan kesalahan data, namun tetap perlu ditangani pada tahap data preparation agar tidak memengaruhi hasil proses pemodelan.

### 5. Visualisasi Data

##### - Distribusi Jumlah Data per Species

![original image](https://cdn.mathpix.com/snip/images/Gt4lANFXJ1tvApce6vXv0vo4hu08VXJb7_bTkbfgn-M.original.fullsize.png)

Grafik bar digunakan untuk melihat jumlah data pada setiap species. Berdasarkan grafik, diketahui bahwa setiap species yaitu Iris-setosa, Iris-versicolor, dan Iris-virginica masing-masing memiliki 50 data. Hal ini menunjukkan bahwa dataset dalam kondisi seimbang (balanced dataset), sehingga tidak terdapat ketimpangan jumlah data antar kelas. Kondisi ini sangat baik untuk proses modeling karena dapat membantu menghasilkan model klasifikasi yang lebih akurat dan tidak bias terhadap kelas tertentu.

##### - Distribusi Data Fitur Numerik pada Dataset Iris

![original image](https://cdn.mathpix.com/snip/images/6u595jYczu5oHxd1yDMvEGkfvzQYsyvFrhOICQnLnGY.original.fullsize.png)

Berdasarkan histogram, dapat disimpulkan bahwa seluruh fitur numerik memiliki distribusi yang bervariasi dan tidak terdapat anomali yang ekstrem. Fitur petal_length dan petal_width menunjukkan pola distribusi yang lebih jelas dalam membedakan kelompok data, sehingga kedua fitur ini sangat berpotensi menjadi fitur penting dalam proses klasifikasi pada tahap modeling. Visualisasi ini membantu dalam memahami karakteristik dan penyebaran data pada tahap Data Understanding dalam metodologi CRISP-DM.

##### - Analisis Penyebaran Data dan Deteksi Outlier Menggunakan Boxplot

![original image](https://cdn.mathpix.com/snip/images/YfGQj5-WjGnpPQF6mWpdtoRSXTPrJIw5sXZ6l769Xo0.original.fullsize.png)

Berdasarkan boxplot, dapat disimpulkan bahwa setiap fitur memiliki penyebaran data yang berbeda. Fitur sepal_width menunjukkan adanya beberapa outlier, sedangkan fitur lainnya memiliki distribusi yang relatif normal tanpa outlier yang signifikan. Fitur petal_length dan petal_width memiliki variasi data yang cukup besar, sehingga berpotensi menjadi fitur penting dalam membedakan species pada tahap modeling. Visualisasi boxplot ini membantu dalam memahami distribusi data dan mendeteksi outlier pada tahap Data Understanding dalam metodologi CRISP-DM.

#### Deskripsi Atribut

| Atribut | Deskripsi | Tipe Data |
| :-- | :-- | :-- |
| sepal_length | Panjang sepal dalam cm | Float |
| sepal_width | Lebar sepal dalam cm | Float |
| petal_length | Panjang petal dalam cm | Float |
| petal_width | Lebar petal dalam cm | Float |
| species | Jenis spesies iris | Kategorikal |

## Statistik Deskriptif Awal

Berdasarkan eksplorasi awal, diperoleh gambaran umum sebagai berikut:

1. Nilai pada atribut pengukuran berada dalam rentang yang wajar dan konsisten.
2. Rata-rata panjang dan lebar petal menunjukkan perbedaan yang cukup jelas antar spesies.
3. Sebaran data relatif seimbang antar kelas, sehingga dataset tidak mengalami masalah ketidakseimbangan kelas yang signifikan.

Statistik deskriptif ini memberikan indikasi bahwa data memiliki pola yang memungkinkan untuk proses klasifikasi.

## Pengecekan Data Duplikat

Hasil pemeriksaan menunjukkan bahwa tidak terdapat data duplikat pada dataset. Setiap baris data merepresentasikan sampel bunga yang unik. Kondisi ini menunjukkan bahwa dataset memiliki kualitas yang baik dan dapat langsung digunakan pada tahap persiapan data tanpa proses penghapusan duplikasi.

# Eksplorasi Data Iris

### Visualisasi

Visualisasi dilakukan untuk memahami pola distribusi data dan hubungan antar variabel. Berdasarkan gambar yang ditampilkan (histogram dan/atau scatter plot), terlihat bahwa distribusi pada variabel petal length dan petal width menunjukkan pemisahan yang cukup jelas antar spesies.

Pada scatter plot antara petal length dan petal width, terlihat bahwa spesies Iris-setosa membentuk kelompok (cluster) yang terpisah secara signifikan dari dua spesies lainnya. Sementara itu, Iris-versicolor dan Iris-virginica memiliki sedikit area tumpang tindih, namun masih dapat dibedakan berdasarkan ukuran petalnya.

Histogram menunjukkan bahwa sebagian besar variabel memiliki distribusi yang relatif normal, meskipun terdapat perbedaan rentang nilai antar spesies. Visualisasi ini membantu dalam memahami bahwa fitur petal lebih efektif dalam membedakan spesies dibandingkan fitur sepal.

Dari visualisasi tersebut dapat disimpulkan bahwa data memiliki pola yang jelas dan tidak tersebar secara acak, sehingga sangat mendukung untuk analisis klasifikasi.

* Sepal

![original image](https://cdn.mathpix.com/snip/images/Y0D4Vc_bKZyv6KgPthBSm8pOPKFWZf-ZcZEN1qaBDpM.original.fullsize.png)

* Petal

![original image](https://cdn.mathpix.com/snip/images/cbXgTI0n7n5F-Hm_QEabnsoY06-1zrCTGJM6BGw0VT4.original.fullsize.png)

### Analisa Korelasi

Analisis korelasi dilakukan untuk mengetahui hubungan antar variabel numerik. Berdasarkan matriks korelasi (heatmap), terlihat bahwa petal length dan petal width memiliki korelasi yang sangat kuat (mendekati +1). Hal ini menunjukkan bahwa semakin panjang petal suatu bunga, maka semakin lebar pula petalnya.

Sepal length memiliki korelasi sedang terhadap petal length dan petal width. Sedangkan sepal width menunjukkan korelasi yang relatif lebih lemah dibandingkan variabel lainnya.

Korelasi yang kuat antara petal length dan petal width menunjukkan bahwa kedua variabel ini memiliki hubungan linier yang signifikan dan berperan penting dalam membedakan spesies. Oleh karena itu, fitur petal menjadi indikator utama dalam proses klasifikasi data Iris.

### Statistik Deskriptif

Berdasarkan hasil statistik deskriptif, diperoleh informasi mengenai nilai rata-rata (mean), standar deviasi, nilai minimum, maksimum, serta kuartil dari masing-masing variabel.

Nilai rata-rata menunjukkan bahwa petal length dan petal width memiliki variasi nilai yang lebih besar dibandingkan sepal width. Standar deviasi yang relatif tinggi pada variabel petal mengindikasikan adanya perbedaan ukuran yang cukup signifikan antar spesies.

Rentang nilai minimum dan maksimum juga memperlihatkan bahwa spesies memiliki karakteristik morfologi yang berbeda, terutama pada ukuran petal. Tidak ditemukan nilai kosong (missing value) dalam dataset, sehingga data dapat digunakan langsung untuk analisis lebih lanjut tanpa proses pembersihan data tambahan.

![original image](https://cdn.mathpix.com/snip/images/HDC--owJyN2brHw80LVv73mBWygr0yxk9lX4t96wFvc.original.fullsize.png)

### Visualisasi Scatter Plot Sepal dan Petal

![original image](https://cdn.mathpix.com/snip/images/wdF8Z79VGVflh2mDtT56v_4f5q5SlGbLHtM778Ut6ac.original.fullsize.png)

![original image](https://cdn.mathpix.com/snip/images/1K4uhfC_chCDhN56N1oE_UogCOe7cSdr_zdgA9iD8zI.original.fullsize.png)
## 6. Mengukur jarak data

### 6.1 Pengertian Similarity dan Dissimilarity

#### 6.1.1 Similarity

Sesuai namanya adalah kemiripan, dalam similarity yang diukur adalah mengukur seberapa mirip dua objek yang sedang disandingkan, semakin besar nilai yang dimiliki maka semakin mirip objek tersebut. Tinggi nilainya berada pada range [0,1]. Semakin tinggi nilai nya, maka semakin mirip objek yang dimaksud

#### 6.1.2 Dissimilarity

Terbanding terbalik dengan similarity, Disisimilarity mengukur seberapa berbeda dua objek yang sedang disandingkan. Nilai nya berawal dari 0, semakin besar nilai yang dimiliki, maka semakin tidak mirip objek tersebut.
Contoh Dissimilarity:

##### 6.1.2.1 Minkowski Distance

Minkowski Distance adalah bentuk generalisasi dari Euclidean dan Manhattan. Artinya Euclidean dan Manhattan adalah bagian dari kelompok

$$
\begin{aligned}
d(i,j) &= \sqrt[h]{|x_{i1}-x_{j1}|^{h} + |x_{i2}-x_{j2}|^{h} + \cdots + |x_{ip}-x_{jp}|^{h}} \\
\\
\text{dengan } 
i &= (x_{i1}, x_{i2}, \dots, x_{ip}), \\
j &= (x_{j1}, x_{j2}, \dots, x_{jp})
\end{aligned}
$$

i = Melambangkan objek ke-i
j = Melambangkan objek ke-j
xi1 = Nilai fitur 1 dari objek ke-i
xj1 = Nilai fitur 1 dari objek ke-j

##### 6.1.2.2 Manhattan Distance

Manhattan Distance mengukur seberapa jauh jarak antara 2 titik dengan menjumlahkan selisih absolut dari koordinatnya. Manhattan distance lebih memliki ketahanan terhadap outlier dibandingkan dengan euclidean distance. Manhattan distance cocok untuk data yang berdimensi tinggi

$$
d(i,j) = |x_{i1} - x_{j1}| + |x_{i2} - x_{j2}| + \cdots + |x_{ip} - x_{jp}|
$$

##### 6.1.2.3 Euclidean Distance

Yang terakhir ada Euclidean Distance, Euclidean digunakan untuk data bertipe numerik meskipun jarak Euclidean sangat umum dalam pengelompokan. Salah satu permasalahan yang ada dalam Euclidean adalahjarak Euclidean sebagai fitur skala terbesar akan mendominasi yang lain. Normalisasi fitur kontinu adalah solusi untuk mengatasi kelemahan ini.

$d(i,j) = \sqrt{(x_{i1} - x_{j1}|^2 +| x_{i2} - x_{j2}|^2 + \ldots + |x_{ip} - x_{jp}|^2})$

### 6.2 Mengukur jarak dissimilarity pada suatu dataset

#### 6.2.1 Mengukur jarak dataset Iris

Pada dataset Iris diatas, tipe data yang dimiliki adalah bernilai numerik, karena hal itu penghitungan jarak untuk tipe data Iris kali ini akan digunakan metode Eucledian Distance

##### 6.2.1.1 Perhitungan manual

Contoh perhitungan secara manual pada baris 2 dan 3 dengan kolom 1:

$$
d(2,1) = \sqrt{(|4{,}9 - 5{,}1|^2 + |3 - 3{,}5|^2 + |1{,}4 - 1{,}4|^2 + |0{,}2 - 0{,}2|^2)}
$$

$$
= \sqrt{0{,}04 + 0{,}25}
$$

$$
= 0{,}538
$$

$$
d(3,1) = \sqrt{(|4{,}7 - 5{,}1|^2 + |3{,}2 - 3{,}5|^2 + |1{,}3 - 1{,}4|^2 + |0{,}2 - 0{,}2|^2)}
$$

$$
= \sqrt{0{,}16 + 0{,}09 + 0{,}01 + 0}
$$

$$
= 0{,}50990
$$

##### 6.2.1.2 Perhitungan Python

Kita juga dapat menggunakan tools seperti python untuk mempermudah perhitungan ini

```
from scipy.spatial.distance import pdist, squareform

X = df.select_dtypes(include=['float64', 'int64'])
dist = pdist(X, metric='euclidean')
distance_matrix = squareform(dist)

distance_df = pd.DataFrame(distance_matrix)
print(distance_df.iloc[:5, :5])
```

Terdapat tambahan library yang digunakan yaitu library scipy untuk mempermudah kita menghitung Euclidean Distance tanpa perlu coding manual. Selanjutnya data ditampilkan hanya sebesar 5x5 dihitung dari yang pertama

Output yang dihasilkan:


|  | 0 | 1 | 2 | 3 | 4 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 0 | 0.000000 | 0.538516 | 0.509902 | 0.648074 | 0.141421 |
| 1 | 0.538516 | 0.000000 | 0.300000 | 0.331662 | 0.608276 |
| 2 | 0.509902 | 0.300000 | 0.000000 | 0.244949 | 0.509902 |
| 3 | 0.648074 | 0.331662 | 0.244949 | 0.000000 | 0.648074 |
| 4 | 0.141421 | 0.608276 | 0.509902 | 0.648074 | 0.000000 |

##### 6.2.1.3 Perhitungan Orange

![original image](https://cdn.mathpix.com/snip/images/9PGgc7ny0uksXf9OeJWUUqJ1W74_wYqqPmGLGPGhBL8.original.fullsize.png)
3 perhitungan diatas, memiliki kesamaan nilai, artinya perhitungan yang dilakukan sudah dilakukan secara benar dan urut

#### 6.2.2 Mengukur jarak dataset Tipe data campuran

Perhitungan

Perhitungan manual antara baris 1 dan 2, lalu baris 1 dan 3: Untuk menghitung jarak pada data campuran, kita perlu menghitungnya pada masing-masing tipe data yang kemudian hasil akhirnya dijumlahkan.

1. Ordinal dan Numerik

Pada kolom parental level of education merupakan fitur dengan tipe ordinal. Pada tipe data ordinal ini sebelum dapat menghitung jarak pada baris 1 dan 2, lalu baris 1 dan 3 perlu melakukan normalisasi, yang setelah dinormalisasi nanti dapat dihitung jaraknya dengan tipe data numerik menggunakan metode Euclidean Distance.

Rumus normalisasi:
⁡
$$
\text{Normalisasi} = \frac{r - \min}{\max - \min}
$$


Contoh proses normalisasi:

$$
1.\ \frac{9-1}{16-1} = \frac{8}{15} = 0.53
$$

$$
2.\ \frac{10-1}{16-1} = \frac{9}{15} = 0.6
$$

Setelah dilakukan normalisasi pada tipe data ordinal, selanjutnya ada proses perhitungan menggunakan metode Euclidean Distance pada fitur numerik.

Data yang digunakan

Perhitungan Jarak
$$
\begin{aligned}
d(1,2) &= \sqrt{(0.53-0.53)^2 + (82-90)^2 + (18-48)^2} \\
&= \sqrt{0 + 64 + 484} \\
&= \sqrt{548} \\
&= 23.409
\end{aligned}
$$

$$
\begin{aligned}
d(1,3) &= \sqrt{(0.6-0.53)^2 + (66-90)^2 + (48-48)^2} \\
&= \sqrt{0.0049 + 576} \\
&= \sqrt{576.0049} \\
&= 24.000102
\end{aligned}
$$

Sehingga diperoleh:

$$
d(1,2) = 23.409
$$

$$
d(1,3) = 24.000102
$$
2. Binary

Perhitungan jarak pada tipe data biner perlu menentukan apakah atribut tersebut simetris atau asimetris, karena rumusnya berbeda.

a. d(1,2)
$$
q = 0,\quad r = 0,\quad s = 0,\quad t = 1
$$

$$
d = \frac{r+s}{q+r+s+t}
$$

$$
d = \frac{0+0}{0+0+0+1} = 0
$$

b. d(1,3)

$$
q = 0,\quad r = 0,\quad s = 0,\quad t = 1
$$

$$
d = \frac{r+s}{q+r+s+t}
$$

$$
d = \frac{0+0}{0+0+0+1} = 0
$$


3. Kategori

Pada tipe data kategori digunakan rumus:

$$
d = \frac{p - m}{p}
$$


dimana

𝑝 = jumlah fitur

𝑚 = jumlah kategori yang sama.

$$
d(1,2) = \frac{1-0}{1} = 1
$$

$$
d(1,3) = \frac{1-1}{1} = 0
$$

Hasil Akhir

$$
d(1,2) = 23.409 + 0.5 + 1 = 24.909
$$

$$
d(1,3) = 24.000102 + 0 + 0 = 24.000102
$$
## 1. Dataset yang Digunakan

Dataset yang digunakan adalah **Boston Housing Dataset** yang diperoleh dari Kaggle. Dataset ini berisi berbagai atribut yang menggambarkan kondisi lingkungan, sosial ekonomi, serta karakteristik rumah di kota Boston.

Dataset ini sering digunakan dalam machine learning untuk memprediksi harga rumah berdasarkan beberapa variabel seperti tingkat kriminalitas, jumlah kamar, dan kondisi lingkungan.

---

## 2. Perhitungan WKNN Manual

### 2.1 Identifikasi Missing Value
Terdapat missing value pada atribut **LSTAT** yang akan diimputasi menggunakan metode WKNN.

---

### 2.2 Normalisasi Data
Data dinormalisasi menggunakan **Min-Max Normalization** agar semua atribut memiliki skala yang sama.

Rumus:

$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

---

### 2.3 Menghitung Similarity (Si)

$$
S_i = \frac{1}{\sum (Y_{ih} - Y_{jh})^2}
$$

Hasil:

| Si |
|----|
| 0.096432567 |
| 0.186117946 |
| 0.251889237 |
| 1.262310597 |

---

### 2.4 Menghitung Penyebut

$$
\sum S_i = 1.796750348
$$

---

### 2.5 Menghitung Pembilang

| Si | Yjh | Si × Yjh |
|---|---|---|
| 0.096432567 | 0.329032258 | 0.031730 |
| 0.186117946 | 1 | 0.186118 |
| 0.251889237 | 0.175806452 | 0.044277 |
| 1.262310597 | 0 | 0 |

$$
\sum (S_i \times Y_{jh}) = 0.262125
$$

---

### 2.6 Hasil Imputasi

$$
\hat{y}_{ih} =
\frac{0.262125}{1.796750348}
$$

$$
\hat{y}_{ih} = 0.14589179
$$

Nilai ini digunakan untuk menggantikan missing value pada atribut **LSTAT**.

---

## 3. Perhitungan WKNN Menggunakan Python

Perhitungan dilakukan menggunakan Python untuk memverifikasi hasil manual.

### 3.1 Import Library
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
```
### 3.2 Membaca Dataset
```
df = pd.read_excel("data.xlsx")
df = df.replace("?", np.nan)
df = df.apply(pd.to_numeric)

print(df)
```
### 3.3 Menentukan Kolom Normalisasi
```
cols_norm = [
'CRIM','INDUS','NOX','RM','AGE',
'DIS','TAX','PTRATIO','B','LSTAT','MEDV'
]
```
### Menghitung Min dan Max
```
min_val = df[cols_norm].min()
max_val = df[cols_norm].max()

print(min_val)
print(max_val)
```
### 3.5 Normalisasi Data
```
df_norm = df.copy()

for col in cols_norm:
    df_norm[col] = (df[col] - min_val[col]) / (max_val[col] - min_val[col])

print(df_norm)
```
### 3.6 Menentukan Data Missing

```
missing_index = df_norm[df_norm['LSTAT'].isna()].index
row_missing = df_norm.loc[missing_index].drop(columns=['LSTAT']).iloc[0]

complete_data = df_norm.dropna()
```
### 3.7 Menghitung Similarity
```
Si = []
Yjh = []

for i in range(len(complete_data)):
    row = complete_data.iloc[i].drop('LSTAT')
    diff = (row_missing - row) ** 2
    similarity = 1 / diff.sum()
    
    Si.append(similarity)
    Yjh.append(complete_data.iloc[i]['LSTAT'])

Si = np.array(Si)
Yjh = np.array(Yjh)

print(Si)
```
OUTPUT
```
[0.09643257 0.18611795 0.25188924 1.2623106]
```
### 3.8 Menghitung Penyebut
```
penyebut = np.sum(Si)
print(penyebut)
```
OUTPUT
```
1.796750347642724
```
### 3.9 Menghitung Pembilang
```
pembilang = np.sum(Si * Yjh)
print(pembilang)
```
OUTPUT
```
0.262125
```
### 3.10 Hasil Imputasi
```
hasil = pembilang / penyebut
print(hasil)
```
OUTPUT
```
0.14589179
```
### 3.11 Kesimpulan
Hasil imputasi menggunakan Python menghasilkan nilai 0.14589179, sama dengan perhitungan manual. Hal ini menunjukkan bahwa proses perhitungan WKNN telah dilakukan dengan benar.
## 2. Normalisasi Data

Normalisasi data merupakan proses mengubah skala nilai pada dataset ke dalam rentang tertentu agar setiap variabel memiliki skala yang sebanding. Tahapan ini biasanya dilakukan pada preprocessing sebelum analisis atau pemodelan machine learning.

Tujuan utama normalisasi adalah menghindari dominasi atribut yang memiliki skala besar sehingga setiap fitur dapat berkontribusi secara seimbang dalam proses perhitungan.

---

### 2.1 Macam-macam Normalisasi Data

#### 1. Min-Max Normalization
Min-Max Normalization digunakan untuk mengubah nilai data ke dalam rentang tertentu, umumnya antara 0 hingga 1.

Rumus:


$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

**Contoh:**

Menggunakan data pada kolom **AGE**:

| AGE |
|----|
| 65.2 |
| 78.9 |
| 45.8 |

Nilai minimum dan maksimum:

- \(x_{min} = 45.8\)
- \(x_{max} = 78.9\)

Perhitungan:

$$
x' = \frac{65.2 - 45.8}{78.9 - 45.8}
$$

$$
x' = \frac{19.4}{33.1}
$$

$$
x' = 0.586
$$

Hasil:

| AGE Asli | AGE Normalisasi |
|---------|----------------|
| 65.2 | 0.586 |
| 78.9 | 1 |
| 45.8 | 0 |

---

#### 2. Z-Score Normalization
Z-Score Normalization adalah metode normalisasi dengan menggunakan rata-rata (mean) dan standar deviasi.

Rumus:

$$
x' = \frac{x - \mu}{\sigma}
$$

**Contoh:**

Data **AGE**:

| AGE |
|----|
| 65.2 |
| 78.9 |
| 45.8 |

##### Menghitung Mean
$$
\mu = \frac{65.2 + 78.9 + 45.8}{3}
$$

$$
\mu = 63.3
$$

##### Menghitung Standar Deviasi
$$
(65.2 - 63.3)^2 = 3.61
$$

$$
(78.9 - 63.3)^2 = 243.36
$$

$$
(45.8 - 63.3)^2 = 306.25
$$

Jumlah:
$$
553.22
$$

$$
\frac{553.22}{3} = 184.41
$$

$$
\sigma = \sqrt{184.41} = 13.58
$$

##### Perhitungan Z-Score
$$
x' = \frac{65.2 - 63.3}{13.58} = 0.14
$$

Hasil:

| AGE Asli | Z-Score |
|---------|--------|
| 65.2 | 0.14 |
| 78.9 | 1.15 |
| 45.8 | -1.29 |

---

#### 3. Decimal Scaling Normalization
Decimal Scaling dilakukan dengan membagi nilai data dengan pangkat 10 berdasarkan jumlah digit terbesar.

Rumus:

$$
x' = \frac{x}{10^j}
$$

**Contoh:**

Data **AGE**:

| AGE |
|----|
| 65.2 |
| 78.9 |
| 45.8 |

Nilai terbesar adalah **78.9**, memiliki 2 digit sebelum desimal → \(j = 2\)

Perhitungan:

$$
x' = \frac{65.2}{10^2} = 0.652
$$

Hasil:

| AGE Asli | Decimal Scaling |
|---------|----------------|
| 65.2 | 0.652 |
| 78.9 | 0.789 |
| 45.8 | 0.458 |

---

### 2.2 Normalisasi Data Menggunakan SKLearn

Selain perhitungan manual, normalisasi juga dapat dilakukan menggunakan library **scikit-learn (sklearn)** untuk mempermudah proses.

Digunakan data pada kolom **AGE**:

| AGE |
|------|
| 65.2 |
| 78.9 |
| 45.8 |

---

#### 1. Min-Max Normalization (sklearn)

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = {'AGE':[65.2,78.9,45.8]}
df = pd.DataFrame(data)

scaler = MinMaxScaler()
df['AGE_MinMax'] = scaler.fit_transform(df[['AGE']])

print(df)
```

OUTPUT:
| AGE  | AGE_MinMax |
| ---- | ---------- |
| 65.2 | 0.586      |
| 78.9 | 1          |
| 45.8 | 0          |

#### 2. Z-Score Normalization (sklearn)
```
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df['AGE_ZScore'] = scaler.fit_transform(df[['AGE']])

print(df)
```
OUTPUT:

| AGE  | AGE_ZScore |
| ---- | ---------- |
| 65.2 | 0.14       |
| 78.9 | 1.15       |
| 45.8 | -1.29      |

#### 3. Decimal Scaling (Python)
```
import numpy as np

data = {'AGE':[65.2,78.9,45.8]}
df = pd.DataFrame(data)

max_value = df['AGE'].max()
j = len(str(int(max_value)))

df['AGE_Decimal'] = df['AGE'] / (10**j)

print(df)
```
OUTPUT:
| AGE  | AGE_Decimal |
| ---- | ----------- |
| 65.2 | 0.652       |
| 78.9 | 0.789       |
| 45.8 | 0.458       |
 #### Kesimpulan
 Hasil normalisasi menggunakan sklearn menunjukkan nilai yang sama dengan perhitungan manual. Hal ini membuktikan bahwa proses normalisasi telah dilakukan dengan benar dan sklearn dapat digunakan untuk mempercepat proses preprocessing data.