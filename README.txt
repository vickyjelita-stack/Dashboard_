PREDICTIVE MAINTENANCE DASHBOARD

ISI PAKET
- app.py: aplikasi dashboard Streamlit
- data_predictive_maintenance.csv: dataset asli dalam format CSV
- pdm_model.joblib: pipeline preprocessing dan Random Forest
- model_config.json: threshold, asumsi biaya, dan daftar fitur
- permutation_importance.csv: hasil interpretasi model
- requirements.txt: dependensi aplikasi

CARA MENJALANKAN
1. Ekstrak ZIP ke satu folder.
2. Instal Python 3.10 atau lebih baru.
3. Buka terminal pada folder hasil ekstrak.
4. Jalankan: pip install -r requirements.txt
5. Jalankan: streamlit run app.py

CATATAN
- Dataset tidak memiliki timestamp kalender. operating_hours digunakan sebagai proxy urutan.
- Biaya FN/FP adalah asumsi simulasi dan perlu divalidasi dengan tim pemeliharaan.
- Hasil model memiliki keterbatasan; gunakan dashboard sebagai alat bantu prioritas pemeriksaan, bukan keputusan otomatis.
