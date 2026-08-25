# Kamera Tabanlı Araç Hız ve Plaka Tespit Sistemi

Bu proje, bir video kaydı üzerinden geçen araçları tespit eden, hızlarını hesaplayan ve plakalarını okuyan görüntü işleme tabanlı bir yazılımdır. Sistem, fiziksel bir radar sensörü kullanmadan sadece pikseller ve zaman hesaplamaları üzerinden çalışır.

## 📁 Projedeki Dosyalar ve Kodun İşleyişi

Proje dosyaları ve kodun arka planda tam olarak yaptığı işlemler aşağıda listelenmiştir:

* **`car_detect_virtual_loop.py` (Ana Çalıştırma Dosyası)**
Sistemi çalıştıran betiktir. Kod çalıştırıldığında sırasıyla şu işlemleri yapar:
1. Videoyu kare kare (frame) okumaya başlar.
2. Her kareyi YOLOv8 yapay zeka modeline göndererek ekrandaki araçların (otomobil, otobüs, kamyon vb.) koordinatlarını (Bounding Box) alır.
3. ByteTrack algoritmasını kullanarak tespit edilen araca bir kimlik (ID) atar ve aracı kareler boyunca aynı ID ile takip eder.
4. Aracın asfalta değdiği alt kenarını referans alır. Bu kenar, yazılımda belirlenen "Giriş" çizgisini (Örn: 510. piksel) geçtiğinde bir kronometre başlatır. "Çıkış" çizgisini (Örn: 590. piksel) geçtiğinde kronometreyi durdurur.
5. İki çizgi arasındaki gerçek mesafe koda sabit olarak girilmiştir. Kod, `Mesafe / Geçen Zaman` formülünü işleterek aracın hızını (km/s) bulur.
6. Aracın plaka bölgesini keser. Görüntü çok bulanıksa işlemciyi yormamak için okumayı pas geçer. Net ise OCR (Karakter Tanıma) algoritması ile plakadaki metni okur ve sonucu ekrana yazar.


* **`batch_evaluate.py` (Toplu Test Dosyası)**
Geliştirme aşamasında sistemin hata payını ölçmek için kullanılır. `videos/` klasöründeki tüm test videolarını sırayla (arayüz açmadan, arka planda) çalıştırır. Her videonun sonunda, kodun ölçtüğü hız ile videonun adında yazan gerçek hızı karşılaştırıp yüzde (%) kaç hata yapıldığını terminale liste halinde basar.
* **`plate_detection/` (Klasörü)**
Sistemin plaka tespit modelini sıfırdan eğitmek (`train_plate.py`) ve eğitilen modeli tek bir görselde test etmek (`test_plate.py`) için yazılmış eğitim kodlarını barındırır. Günlük kullanımda bu klasör çalıştırılmaz.
* **`requirements.txt`**
Kodun çalışması için bilgisayara yüklenmesi gereken Python kütüphanelerinin (OpenCV, ultralytics, numpy vb.) listesidir.

---

## 🚀 Kurulum ve Çalıştırma

**1. Gerekli Kütüphaneleri Yükleyin:**
Projeyi indirdikten sonra terminali açın ve şu komutu çalıştırarak altyapıyı kurun:

```bash
pip install -r requirements.txt

```

**2. Sistemi Çalıştırın:**
Ana kod dosyasını ve işlemek istediğiniz videonun yolunu yazarak sistemi başlatın:

```bash
python car_detect_virtual_loop.py test_videosu.mp4

```

*(Komut çalıştırıldığında bir pencere açılır. Araçlar yeşil kutu içine alınır, hesaplanan hız ve okunan plaka aracın üzerinde canlı olarak gösterilir. Hız sınırını aşan araçlar için kırmızı bir uyarı metni belirir.)*

**3. Hata Payı Raporu Alın (Opsiyonel):**
Birden fazla test videosunun olduğu durumlarda toplu sonuç almak için şu komutu kullanın:

```bash
python batch_evaluate.py

```

---

## 📊 Performans ve Kısıtlamalar

* **Hız Ölçümü:** Sistem, önceden ölçülmüş referans mesafeleri kullandığı için %2 ile %5 arasında düşük bir hata payıyla hız hesaplar.
* **FPS Sınırı:** Kod, pikseller arasındaki geçiş süresini saydığı için videonun FPS (Saniyedeki Kare Sayısı) değerine duyarlıdır. 30 FPS kameralarda, araçlar çok yüksek hızlara (100+ km/h) çıktığında kare atlamalarından dolayı hız ölçümünde ufak sapmalar görülebilir. Kamera FPS değeri arttıkça ölçüm hassasiyeti de artar.
