Celestial Object Classification with SDSS DR19

Automated classification of astronomical objects using machine learning and the Sloan Digital Sky Survey Data Release 19

📋 Overview
This repository contains the final project for CSE418 Data Mining course (Fall 2025-2026) at Aydın Adnan Menderes University. The project focuses on automated classification of celestial objects into three categories:

Stars ⭐
Galaxies 🌀
Quasars (QSO) 💫

With the exponential growth of astronomical data, manual classification by astronomers is no longer feasible. This study follows the KDD (Knowledge Discovery in Databases) process to build a robust classification system using machine learning algorithms.
🎯 Key Features
Data Acquisition

Constructed a high-quality dataset of 10,000 samples
Custom SQL queries joining PhotoObj and SpecObj tables from SDSS SkyServer
SDSS Data Release 19 (DR19)

Feature Engineering

Color Indices: Calculated astrophysical color indices (e.g., u-g, g-r, r-i, i-z) to capture spectral shapes
Coordinate Transformation: Converted spherical coordinates (ra, dec) into 3D Cartesian space (x, y, z)
Redshift Derivatives: Derived features such as redshift_sq (z²) and redshift_snr

Feature Selection

Used ANOVA F-test (SelectKBest) to identify the 12 most discriminative attributes
Redshift ranked as the most critical feature for classification

Exploratory Data Analysis (EDA)

Class Distribution:

Galaxy: 54.7%
Star: 33.4%
QSO: 11.9%


Visualized redshift distribution showing Quasars clustering at significantly higher redshift levels

🛠️ Tech Stack

Language: Python 3.8+
Libraries:

Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn


Framework: CRISP-DM / KDD Process

📊 Results & Performance
Models were evaluated using an 80/20 train/test split and 5-Fold Cross-Validation. Gradient Boosting emerged as the top performer due to its ability to handle non-linear decision boundaries in overlapping feature spaces.
AlgorithmTest AccuracyF1-Score (Weighted)Gradient Boosting98.20%0.9819Random Forest98.15%0.9814SVM97.65%0.9761KNN97.35%0.9731Naive Bayes95.60%0.9560
Key Findings

Tree-based models (Random Forest and Gradient Boosting) significantly outperformed linear models in regions where Galaxies and Quasars overlap in color space
Redshift proved to be the most discriminative feature for separating object classes
The system achieved high accuracy despite class imbalance

👥 Contributors

Gökay Sepet - 201805068
Cemre Polat - 211805054
Özcan Erdem Tosun - 231805003

🎓 Institution
Aydın Adnan Menderes University
Faculty of Engineering
Computer Engineering Department

🌍 Turkish Version / Türkçe Versiyon
🌌 Gök Cisimlerinin Sınıflandırılması (SDSS DR19)

SDSS Veri Sürümü 19 ve makine öğrenmesi kullanarak astronomik nesnelerin otomatik sınıflandırılması

📋 Proje Özeti
Bu depo, Aydın Adnan Menderes Üniversitesi Mühendislik Fakültesi CSE418 Veri Madenciliği dersi (Güz 2025-2026) final projesini içermektedir. Proje, makine öğrenmesi algoritmalarını kullanarak gök cisimlerini otomatik olarak sınıflandırmayı amaçlar:

Yıldızlar ⭐
Galaksiler 🌀
Kuasarlar (QSO) 💫

Astronomik veri miktarındaki devasa artış nedeniyle, gök cisimlerinin manuel olarak sınıflandırılması artık mümkün değildir. Bu çalışma, Sloan Dijital Gökyüzü Taraması (SDSS) Veri Sürümü 19 (DR19) kullanarak sağlam bir sınıflandırma sistemi oluşturmak için KDD (Veritabanlarında Bilgi Keşfi) sürecini takip etmektedir.
🎯 Temel Özellikler
Veri Edinme

SDSS SkyServer üzerinden PhotoObj ve SpecObj tablolarını birleştiren özel SQL sorguları
10.000 örnekten oluşan yüksek kaliteli veri seti

Özellik Mühendisliği

Renk İndeksleri: Tayfsal şekilleri yakalamak için astrofiziksel renk indeksleri (u-g, g-r, vb.)
Koordinat Dönüşümü: Küresel koordinatlar (ra, dec) → 3D Kartezyen koordinatlar (x, y, z)
Redshift Türevleri: Kırmızıya kayma karesi (z²) gibi özellikler

Özellik Seçimi

ANOVA F-testi kullanılarak en etkili 12 değişken belirlendi
Redshift en kritik özellik olarak tespit edildi

📊 Sonuçlar
Beş farklı algoritma karşılaştırılmış ve en yüksek başarıyı %98.20 doğruluk oranı ile Gradient Boosting algoritması göstermiştir.
AlgoritmaTest DoğruluğuF1-Skoru (Ağırlıklı)Gradient Boosting%98.200.9819Random Forest%98.150.9814SVM%97.650.9761KNN%97.350.9731Naive Bayes%95.600.9560
Önemli Bulgular

Ağaç tabanlı modeller (Random Forest ve Gradient Boosting), özellikle Galaksi ve Kuasarların renk uzayında çakıştığı bölgelerde doğrusal modellere göre çok daha iyi performans sergilemiştir
Redshift, nesne sınıflarını ayırt etmede en önemli özellik olmuştur

🛠️ Teknoloji Yığını

Dil: Python 3.8+
Kütüphaneler: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
Süreç: CRISP-DM / KDD

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

Sloan Digital Sky Survey (SDSS) for providing the astronomical data
Aydın Adnan Menderes University for academic support
