# Makalah Algeo — Clustering & Analisis Stabilitas Harga Komoditas

Implementasi untuk makalah **IF2123 Linear Algebra & Geometry**: *clustering* komoditas berdasarkan fitur vektor dari log-return harga, lalu analisis stabilitas dinamika klaster menggunakan eigenvalue (radius spektral).

## Struktur
- `main.py` — pipeline preprocessing → ekstraksi fitur → K-Means → analisis stabilitas (VAR(1)) → visualisasi
- `data.csv` — data harga komoditas
- `hasil_analisis.png` — output grafik hasil analisis (dibuat setelah program dijalankan)

## Format Data
Program membaca `data.csv` dengan kolom minimal:
- `tanggal` (contoh: `2025-09-01`)
- `komoditas` (nama komoditas)
- `harga_idr` (harga dalam Rupiah, numerik)

Kolom lain (mis. `pasar`, `pedagang`, `satuan`) boleh ada dan akan diabaikan untuk perhitungan inti.

## Cara Menjalankan
Pastikan dependensi terpasang: `numpy`, `pandas`, `matplotlib`, `scikit-learn`.

Jalankan:
```bash
python main.py
```

## Output
- Ringkasan hasil per klaster di terminal (termasuk radius spektral $\rho$ dan status stabil/tidak stabil).
- File gambar: `hasil_analisis.png`.

## Author
Niko Samuel Simanjuntak — **13524029**
