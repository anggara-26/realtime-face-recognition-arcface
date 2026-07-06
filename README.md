# Pengenalan Wajah Real-Time untuk Identifikasi Buronan (Prototype)

Implementasi kerja dari makalah *"Pengenalan Wajah Real-Time untuk
Identifikasi Buronan Berbasis Deteksi Wajah dan Embedding ArcFace"*
(Kelompok 7, Universitas Mercu Buana). Berisi dua bagian:

1. **`faceid/engine/`** — model AI-nya sendiri: modul Python murni (tidak
   bergantung pada Django) yang mengimplementasikan pipeline deteksi +
   embedding + cosine matching, dengan dua backend yang bisa ditukar:
   - `arcface` — SCRFD (deteksi) + ArcFace 512-D (InsightFace `buffalo_l`), model optimasi.
   - `dlib` — HOG (deteksi) + ResNet 128-D (`face_recognition`), baseline.
2. **Aplikasi web Django** (`config/`, `faceid/`) — purwarupa: pendaftaran
   satu foto lewat webcam/unggahan dan pengenalan real-time dengan kotak +
   tag nama di atas video, sesuai BAB III.3.8 & BAB IV.4.5 pada makalah.

Selain itu ada `scripts/` yang mereproduksi keempat eksperimen BAB IV
(ROC/EER, robustness terhadap degradasi CCTV, trade-off kecepatan, dan studi
one-shot + test-time augmentation) menggunakan modul engine yang sama.

## Setup

Windows PowerShell, Python 3.11+ (diuji dengan 3.14):

```powershell
python -m venv venv
venv\Scripts\python -m pip install --upgrade pip
venv\Scripts\python -m pip install -r requirements.txt
```

> Catatan: `dlib` di Windows biasanya butuh CMake + Visual Studio Build
> Tools jika tidak ada wheel prebuilt untuk versi Python kamu. Jika instalasi
> `dlib`/`face-recognition` gagal, hapus dua baris itu dari
> `requirements.txt` dan cukup pakai backend `arcface` (yang tidak butuh
> kompilasi apa pun) — UI akan otomatis menampilkan pesan error yang jelas
> jika baseline dlib dipilih tapi belum terpasang.

Migrasi database (SQLite, satu file lokal):

```powershell
venv\Scripts\python manage.py migrate
```

Jalankan server:

```powershell
venv\Scripts\python manage.py runserver
```

Buka `http://127.0.0.1:8000/` di browser yang mengizinkan akses webcam.
Model ArcFace (`buffalo_l`, ~300MB) diunduh otomatis oleh InsightFace saat
backend `arcface` pertama kali dipakai, dan disimpan ke
`~/.insightface/models/`.

## Alur pemakaian prototipe

1. **Daftarkan identitas** (panel kanan): isi nama, ambil snapshot dari
   webcam atau unggah 1+ foto, lalu klik "Daftarkan".
   - Centang *Test-Time Augmentation* untuk mensimulasikan strategi "1 foto
     + TTA" pada BAB IV.4.4 (flip horizontal, rotasi kecil, perubahan
     kecerahan dirata-ratakan menjadi satu embedding).
   - Unggah beberapa foto sekaligus untuk mensimulasikan strategi "3 foto".
2. **Mulai Pengenalan** (panel kiri): sistem mengambil frame dari webcam
   secara berkala, mengirimkannya ke `/api/recognize/`, lalu menggambar
   kotak hijau (dikenal, dengan nama + skor) atau merah (`Tak Dikenal`) di
   atas video — persis seperti Gambar 4.5 pada makalah.
3. Ganti dropdown **Backend / Model** untuk membandingkan baseline `dlib`
   vs optimasi `arcface`; ambang cosine similarity default per backend
   mengikuti temuan EER pada BAB IV.4.1 dan bisa digeser manual.

## Struktur API

| Method | Endpoint             | Keterangan                                   |
|--------|----------------------|-----------------------------------------------|
| POST   | `/api/enroll/`        | multipart: `name`, `engine`, `tta`, `images[]` |
| POST   | `/api/recognize/`     | JSON: `image` (data URL), `engine`, `threshold`|
| GET    | `/api/persons/`       | `?engine=arcface\|dlib`                        |
| DELETE | `/api/persons/<id>/`  | Hapus satu identitas dari galeri               |

## Reproduksi eksperimen BAB IV

```powershell
venv\Scripts\python scripts\prepare_demo_data.py   # unduh subset kecil LFW via scikit-learn
venv\Scripts\python scripts\eval_roc_eer.py
venv\Scripts\python scripts\eval_robustness.py
venv\Scripts\python scripts\eval_speed.py
venv\Scripts\python scripts\eval_oneshot_tta.py
```

Setiap skrip memakai `faceid/engine/` secara langsung (tanpa perlu Django
berjalan) dan menyimpan grafik ke `reports/*.png`, meniru gaya figur pada
Gambar 4.1&ndash;4.4 di makalah. `data/demo/<nama_orang>/*.jpg` adalah
struktur dataset yang dipakai; `prepare_demo_data.py` akan mengisinya secara
otomatis dari LFW jika folder tersebut masih kosong.

## Struktur proyek

```
config/            pengaturan proyek Django
faceid/
  engine/           model AI: interface + backend arcface & dlib (pure Python)
  imaging.py        decode gambar, augmentasi TTA, degradasi CCTV (pure Python)
  models.py         Person (galeri satu embedding per identitas)
  views.py          API enroll/recognize/persons
  templates/, static/  UI webcam + overlay kotak deteksi
scripts/            reproduksi eksperimen BAB IV (ROC/EER, robustness, speed, one-shot)
data/demo/          dataset demo kecil (dibuat otomatis)
reports/            output grafik evaluasi
```
