# AI-Based Vehicle Speed & Plate Detection System
<img width="1268" height="676" alt="image" src="https://github.com/user-attachments/assets/283666f2-0ca0-4549-96fe-67019984c2ec" />

## Bu proje ne yapar?

Bir yol/kavşak videosu izler, videodaki araçları tespit eder, her aracın
**km/h cinsinden hızını** hesaplar ve **plakasını okur**. Fiziksel bir radar
cihazı kullanılmaz — sadece kameradan alınan görüntü işlenerek hız hesaplanır.

Ekranda şöyle bir sonuç görürsün:

```
ID 3: 87.4 km/h | 34ABC123 !!! LIMIT ASIMI !!!
```

## Nasıl çalışır? 
1. **Aracı bul ve takip et:** Videodaki her karede, YOLO adlı bir yapay zeka
   modeli araçları (kutu çizerek) tespit eder. Aynı araç farklı karelerde
   aynı "ID" numarasıyla takip edilir, böylece "bu az önce gördüğümüz araç mı"
   sorusuna cevap verilmiş olur.

2. **Hızı ölç:** Ekranda hayali iki çizgi vardır: bir **giriş çizgisi**, bir
   de **çıkış çizgisi**. Araç giriş çizgisini geçtiği an bir kronometre
   başlar, çıkış çizgisini geçtiği an durur. Bu iki çizgi arasındaki gerçek
   mesafe (metre cinsinden, önceden ölçülmüştür) geçen süreye bölünerek hız
   hesaplanır:

   ```
   hız = mesafe (metre) / geçen süre (saniye)
   ```

   Buna projede **"Sanal Radar" (Virtual Loop)** deniyor, çünkü fiziksel bir
   radar/sensör yok, çizgiler sadece video üzerinde hayali olarak tanımlanmış.

3. **Plakayı oku:** Araç tespit edilen kutunun içinde, ikinci bir YOLO modeli
   plakanın tam olarak nerede olduğunu bulur. Görüntü bulanıksa (araç hızlı
   gittiği için bulanıklaşmış olabilir) o kare atlanır, kaynak boşa
   harcanmaz. Net görüntüler bir OCR (görüntüden yazı okuma) modeline
   verilir ve plaka metni elde edilir.

4. **En doğru plakayı seç:** Bir araç ekranda kaldığı süre boyunca onlarca
   kare görülür ve her karede plaka farklı okunabilir (OCR hatasız değildir).
   Bu yüzden tüm okumalar toplanır ve harf harf **"çoğunluk oylaması"**
   yapılır: her pozisyonda en sık çıkan harf/rakam nihai plaka olarak kabul
   edilir.

5. **Sonucu göster:** Hız, plaka ve (varsa) hız limiti aşımı uyarısı ekrana
   ve terminale yazdırılır.

---

## Dosyalar — her biri ne işe yarıyor?

### 🟢 `detect_speed.py` — PROJENİN ANA DOSYASI
Buradan çalıştırırsın. Bir video dosyası verirsin, o video oynatılır (veya
arka planda işlenir), ekranda araçların hızı ve plakası gösterilir. Yukarıda
anlatılan 5 adımın hepsini birbirine bağlayan dosya budur. Diğer tüm
dosyalar (`config.py`, `virtual_loop.py`, `plate_reader.py`) bu dosyanın
içeride kullandığı "yardımcı parçalardır".

**Nasıl çalıştırılır:**
```bash
python detect_speed.py videos/ornek_video.mp4
```

### ⚙️ `config.py` — AYARLAR
Kod içinde hiçbir "sihirli sayı" olmasın diye, değiştirilmesi muhtemel tüm
ayarlar bu dosyada toplanmıştır: hangi model dosyası kullanılacak, giriş/çıkış
çizgileri video üzerinde hangi pikselde, hız limiti kaç km/h, vs.

### 🎯 `virtual_loop.py` — HIZ HESAPLAMA MANTIĞI
Yukarıdaki 2. adımı (Sanal Radar) yapan dosya. İçinde `VirtualLoop` adında
bir yapı var: her araç ID'si için "bu araç giriş çizgisini ne zaman geçti,
çıkış çizgisini ne zaman geçti" bilgisini tutar ve ikisinden hızı hesaplar.

### 🔤 `plate_reader.py` — PLAKA OKUMA MANTIĞI
Yukarıdaki 3. ve 4. adımı (plaka bulma, netlik kontrolü, OCR, çoğunluk
oylaması) yapan dosya.

### 📊 `batch_evaluate.py` — TOPLU TEST ARACI
`detect_speed.py` tek bir videoyu çalıştırır. Bu dosya ise `videos/`
klasöründeki **tüm videoları arka arkaya, otomatik olarak** çalıştırır ve
sistemin ne kadar doğru sonuç verdiğini bir tabloda özetler (gerçek hız ile
ölçülen hız arasındaki fark, yüzde olarak).

Bu dosya, sistemi elle tek tek test etmek yerine "genel başarı oranım kaç?"
sorusuna hızlıca cevap almak için kullanılır. Ana sistemin çalışması için
**gerekli değildir**, sadece geliştirme/test aşamasında işe yarar.

**Nasıl çalıştırılır:**
```bash
python batch_evaluate.py
```

### 📂 `plate_detection/` klasörü — MODEL EĞİTİM ARAÇLARI
Bu klasördeki dosyalar ana sistemin **parçası değildir**. Plaka tespit
modelini (YOLO) sıfırdan eğitmek veya test etmek için, sadece bir kere,
model geliştirme aşamasında kullanılır.

- **`train_plate.py`** → Plaka tespit modelini bir veri setiyle eğitir. Bu
  script çalıştırıldığında ortaya çıkan model dosyası (`best.pt`),
  `detect_speed.py`'nin plaka bulmak için kullandığı modeldir.
- **`test_plate.py`** → Eğitilmiş modelin düzgün çalışıp çalışmadığını tek
  bir resim üzerinde hızlıca görsel olarak kontrol etmek için kullanılır.

---

## Kurulum ve çalıştırma

**1. Gerekli kütüphaneleri kur:**
```bash
pip install -r requirements.txt
```

**2. Tek bir videoyu test et:**
```bash
python detect_speed.py videos/ornek_video.mp4
```

**3. Tüm videoları toplu test et (sistemin genel doğruluğunu görmek için):**
```bash
python batch_evaluate.py
```

---

## Kullanılan teknolojiler

| Teknoloji | Ne için kullanılıyor |
|---|---|
| **YOLOv8n** | Araçları ve plakaları tespit etmek (görüntüdeki nesneleri bulmak) |
| **ByteTrack** | Aynı aracı farklı karelerde aynı ID ile takip etmek |
| **fast-plate-ocr** | Plaka görselinden yazıyı (harf/rakam) okumak |
| **OpenCV** | Video okuma, görüntü işleme, ekrana çizim yapma |

---

## Veri Seti

Test videoları, açık kaynaklı **VS13 (Vehicle Speed 13)** veri setinden
alınmıştır: [slobodan.ucg.ac.me/science/vs13](https://slobodan.ucg.ac.me/science/vs13/)

Sadece **Peugeot 3008** modelinin farklı hızlarda (40–100 km/h) çekilmiş
videoları kullanılmıştır, böylece sonuçlar standart bir şekilde
karşılaştırılabilir.

---

## Bilinen sınırlamalar

- Sistem hesabı **video kaç kare/saniye (FPS) çekildiyse** o hassasiyette
  yapabilir. 30 FPS'lik bir videoda, çok hızlı giden bir araç iki kare
  arasında ~1 metre yol alabilir; bu da küçük ölçüm sapmalarına yol açabilir.
  60-120 FPS kamera kullanmak bu sapmayı azaltır.
- Genel hız ölçüm hatası **%2-%5** aralığındadır 
