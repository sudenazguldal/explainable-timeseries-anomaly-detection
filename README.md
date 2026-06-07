# From Black-Box to Explainability: Probabilistic Automata for Time Series Analysis

## Grup Bilgileri

| Alan | Bilgi |
|---|---|
| Grup No | 29 |
| Öğrenci 1 | Hüseyin Erekmen - 251307099 |
| Öğrenci 2 | Sudenaz Güldal - 221307029 |
| Proje konusu | Zaman serisi anomali tespiti için Deep Learning ve Probabilistic Automata karşılaştırması |
| Kullanılan veri setleri | SKAB, BATADAL |
| Veri kaynağı | Veri setleri resmi kaynaklardan alınan ham veri dosyaları üzerinden hazırlanmıştır. |

Bu repository, endüstriyel zaman serisi verilerinde anomali tespiti yapmak için iki farklı yaklaşımı karşılaştırır:

1. **Black-box deep learning modelleri:** LSTM ve 1D-CNN.
2. **Açıklanabilir probabilistic automata modeli:** PCA, PAA, SAX, sliding-window pattern çıkarımı, transition probability ve Levenshtein tabanlı unseen pattern yönetimi.

Projenin temel amacı sadece en yüksek accuracy değerini bulmak değildir. Asıl amaç, derin öğrenme modellerinin yüksek tahmin gücü ile otomata modelinin adım adım açıklanabilir karar üretme kapasitesini aynı veri setleri üzerinde karşılaştırmaktır.

## Kısa Sonuç

SKAB veri setinde deep learning modelleri açık şekilde daha yüksek F1-score üretmiştir. En iyi sonuç `0.850 ± 0.082` F1 ile LSTM modelinden gelmiştir. 1D-CNN modeli de SKAB üzerinde güçlüdür ve `0.798 ± 0.120` F1 elde etmiştir.

BATADAL veri setinde modeller daha fazla zorlanmıştır. LSTM `0.227 ± 0.312` F1 üretirken, 1D-CNN modeli yüksek accuracy değerine rağmen anomalileri yakalayamadığı için F1-score `0.000` kalmıştır. Bu durum, dengesiz anomali verilerinde accuracy metriğinin tek başına yanıltıcı olabileceğini göstermektedir.

Automata modeli sabit parametrelerle düşük F1 üretmiştir; ancak `window_size` ve `alphabet_size` ayarları optimize edildiğinde performans belirgin şekilde artmıştır:

| Veri Seti | Sabit Automata F1 | En İyi Automata F1 | En İyi Parametre |
|---|---:|---:|---|
| BATADAL | 0.182 | 0.257 | `window_size=6`, `alphabet_size=5` |
| SKAB | 0.077 ± 0.030 | 0.438 ± 0.030 | `window_size=6`, `alphabet_size=6` |

Bu sonuç şunu gösterir: Automata modeli deep learning kadar yüksek skor üretmese bile, kararlarını state, transition probability, unseen mapping ve confidence score üzerinden açıklayabildiği için yorumlanabilirlik açısından güçlüdür.

## Projenin Ne Yaptığı

Bu proje bir **zaman serisi anomali tespiti** projesidir. Zaman serisi, ölçümlerin zaman boyunca sırayla kaydedildiği veri tipidir. Örneğin bir makinedeki sıcaklık, basınç, elektrik akımı veya titreşim değerleri her saniye ölçülürse elimizde zaman serisi oluşur.

Anomali ise sistemin normal davranışından sapmasıdır. Endüstriyel sistemlerde anomali; arıza, saldırı, sensör bozulması, valf problemi, basınç kaçağı veya beklenmeyen operasyonel davranış anlamına gelebilir.

Projede iki ayrı soru cevaplanmaya çalışılmıştır:

1. Deep learning modelleri bu anomalileri ne kadar iyi yakalıyor?
2. Automata modeli daha düşük skor alsa bile kararını daha anlaşılır şekilde açıklayabiliyor mu?

## Kullanılan Veri Setleri

### SKAB

SKAB, endüstriyel bir test ortamından alınmış sensör kayıtlarını içerir. Bu projede yalnızca `valve1` ve `valve2` klasörleri kullanılmıştır. Her CSV dosyasının hangi kaynaktan geldiğini takip etmek için `source_group` ve `source_file` metadata sütunları eklenmiştir. Bu metadata sütunları model girdisine verilmemiştir; yalnızca split ve veri takibi için kullanılmıştır.

| Özellik | Değer |
|---|---:|
| Satır sayısı | 22472 |
| Feature sayısı | 8 |
| Target sütunu | `anomaly` |
| Normal örnek sayısı | 14646 |
| Anomali örnek sayısı | 7826 |
| Kullanılan source file sayısı | 20 |

SKAB üzerinde kullanılan sensör sütunları:

| Sütun | Anlamı |
|---|---|
| `Accelerometer1RMS` | Birinci ivme sensörünün RMS titreşim değeri. Makinedeki titreşim şiddetini özetler. |
| `Accelerometer2RMS` | İkinci ivme sensörünün RMS titreşim değeri. Farklı noktadaki titreşim davranışını gösterir. |
| `Current` | Elektrik akımı. Motor veya sistemin çektiği akım değişimleri arıza/anomali göstergesi olabilir. |
| `Pressure` | Basınç değeri. Valf veya akış problemlerinde değişebilir. |
| `Temperature` | Sıcaklık değeri. Isınma veya soğuma davranışı sistem sağlığını gösterebilir. |
| `Thermocouple` | Termokupl sıcaklık sensörü ölçümü. Sıcaklığı farklı bir sensör tipiyle takip eder. |
| `Voltage` | Elektrik gerilimi. Sistem besleme veya çalışma koşullarını temsil eder. |
| `Volume Flow RateRMS` | Hacimsel akış hızının RMS değeri. Sistemdeki akış davranışını özetler. |

SKAB için random satır karıştırma yapılmamıştır. Aynı CSV dosyasından gelen örneklerin hem train hem test içine düşmesini engellemek için `source_file` tabanlı group split kullanılmıştır. Bu seçim veri sızıntısını azaltır.

### BATADAL

BATADAL, su dağıtım sistemi üzerinde saldırı/anomali tespiti için kullanılan bir veri setidir. Bu projede `Training Dataset 2` kullanılmıştır. `DATETIME` sütunu yalnızca zaman sırasını korumak için kullanılmış, model girdisine dahil edilmemiştir. Hedef sütun `ATT_FLAG` olarak alınmıştır.

| Özellik | Değer |
|---|---:|
| Satır sayısı | 4177 |
| Feature sayısı | 43 |
| Target sütunu | `ATT_FLAG` |
| Time sütunu | `DATETIME` |
| Normal örnek sayısı | 3958 |
| Anomali örnek sayısı | 219 |
| Train satır sayısı | 2506 |
| Validation satır sayısı | 835 |
| Test satır sayısı | 836 |

BATADAL çok dengesiz bir veri setidir. Normal örnek sayısı anomali örnek sayısından çok daha fazladır. Bu yüzden model sürekli "normal" dese bile accuracy yüksek çıkabilir. Bu projede BATADAL 1D-CNN sonucunda görülen temel problem de budur: accuracy yaklaşık `0.901` olmasına rağmen model anomalileri yakalayamadığı için F1-score `0.000` olmuştur.

BATADAL için veri kronolojik sırayla ayrılmıştır:

```text
%60 train -> %20 validation -> %20 test
```

Random split kullanılmamıştır, çünkü zaman serilerinde gelecekteki verinin geçmiş eğitim verisine sızması gerçekçi olmayan sonuçlar doğurur.

## Terim Sözlüğü

| Terim | Bu projedeki anlamı |
|---|---|
| Time series | Zamana göre sıralı sensör ölçümleri. |
| Feature | Modele verilen giriş değişkeni. Örneğin basınç, sıcaklık, akım. |
| Target / Label | Modelin tahmin etmeye çalıştığı sınıf. Bu projede `0=normal`, `1=anomaly`. |
| Anomaly | Normal sistem davranışından sapma. |
| Train set | Modelin öğrendiği veri bölümü. |
| Validation set | Parametre/threshold seçmek için kullanılan ara veri bölümü. |
| Test set | En son performans ölçmek için ayrılan veri bölümü. |
| Data leakage | Test bilgisinin eğitim aşamasına sızması. Sonuçları olduğundan iyi gösterir. |
| Accuracy | Toplam doğru tahmin oranı. Dengesiz veri setlerinde tek başına yeterli değildir. |
| Precision | Model "anomali" dediğinde ne kadar haklı olduğunu gösterir. |
| Recall | Gerçek anomalilerin ne kadarının yakalandığını gösterir. |
| F1-score | Precision ve recall dengesini tek değerde özetler. Anomali tespitinde çok önemlidir. |
| LSTM | Zaman bağımlılıklarını öğrenebilen recurrent deep learning modeli. |
| 1D-CNN | Zaman pencereleri üzerinde lokal örüntüleri yakalayan convolutional model. |
| Sliding window | Zaman serisini sabit uzunluklu ardışık parçalara ayırma yöntemi. |
| PCA | Çok boyutlu veriyi daha az boyuta indirir. Automata tarafında PC1 kullanılmıştır. |
| PAA | Zaman serisini segmentlere bölüp her segmentin ortalamasını alır. |
| SAX | Sayısal zaman serisini sembollere çevirir. Örneğin `a`, `b`, `c`. |
| Alphabet size | SAX sırasında kaç farklı sembol kullanılacağını belirler. |
| Window size | Kaç sembolün bir state/pattern oluşturacağını belirler. |
| State | Automata içindeki sembolik durum. Örneğin `abca`. |
| Transition | Bir state'ten başka bir state'e geçiş. |
| Transition probability | Bir geçişin eğitim verisinde ne kadar olası olduğunu gösterir. |
| Transition density | Olası tüm geçişlerin ne kadarının gerçekten gözlendiğini gösteren yoğunluk. |
| Unseen pattern | Eğitimde hiç görülmeyen ama test sırasında gelen pattern. |
| Levenshtein distance | İki sembolik dizinin birbirine ne kadar benzediğini ölçen düzenleme mesafesi. |
| Confidence score | Automata kararının olasılık tabanlı güven değeri. |

## Metodoloji

### Deep Learning Pipeline

Deep learning tarafında iki model kullanılmıştır:

| Model | Neden kullanıldı? |
|---|---|
| LSTM | Zaman serilerinde geçmiş ölçümlerin gelecekteki davranışa etkisini öğrenebilir. |
| 1D-CNN | Zaman pencereleri içindeki kısa lokal örüntüleri yakalayabilir. |

Pipeline akışı:

```text
Ham sensör verisi
-> target/metadata ayrımı
-> train üzerinde normalization fit
-> validation/test üzerinde sadece transform
-> sliding-window sequence üretimi
-> LSTM ve 1D-CNN eğitimi
-> accuracy, precision, recall, F1, confusion matrix, PR/ROC curve çıktıları
```

Deep learning ayarları:

| Ayar | Değer |
|---|---:|
| Sequence length | 32 |
| Batch size | 32 |
| Max epoch | 50 |
| Early stopping patience | 5 |
| Random seeds | 42, 123, 2026, 7, 999 |

SKAB tarafında 5 fold ve 5 seed kullanıldığı için her model 25 run ile özetlenmiştir. BATADAL tarafında zaman sıralı tek split ve 5 seed kullanıldığı için her model 5 run ile özetlenmiştir.

### Automata Pipeline

Automata modeli deep learning gibi doğrudan çok boyutlu sensör matrisini öğrenmez. Önce veri sembolik bir zaman serisine dönüştürülür.

```text
Çok değişkenli sensör verisi
-> standard scaling
-> PCA ile PC1 çıkarımı
-> PAA ile segment ortalamaları
-> SAX ile sembolik dizi
-> sliding-window pattern çıkarımı
-> pattern = state
-> transition probability hesaplama
-> düşük olasılıklı geçiş = anomaly adayı
```

Automata modelinin temel varsayımı:

```text
Eğitimde sık görülen transition -> normal davranış
Eğitimde az görülen veya hiç görülmeyen transition -> anomali adayı
```

Sabit automata parametreleri:

| Parametre | Değer |
|---|---:|
| `window_size` | 4 |
| `alphabet_size` | 3 |
| `paa_segments` | 256 |
| `fallback_probability` | 0.000001 |
| `anomaly_threshold` | 0.05 |

Parametre analizi için denenen değerler:

```text
window_size:   3, 4, 5, 6
alphabet_size: 3, 4, 5, 6
```

### Unseen Pattern Handling

Test sırasında eğitimde hiç görülmeyen bir pattern gelirse model hata vermemelidir. Bu durumda Levenshtein distance kullanılarak en yakın bilinen state bulunur.

Örnek:

```text
Gelen unseen pattern: ccca
Bilinen en yakın state: acca
Edit distance: 1
```

Model daha sonra bu eşleme üzerinden transition probability ve karar üretir. Bu sayede automata modeli yalnızca tahmin üretmez, tahminin nedenini de açıklayabilir.

## Deney Tasarımı

| Başlık | Uygulanan karar |
|---|---|
| Veri seti sayısı | 2 veri seti: SKAB ve BATADAL |
| DL model sayısı | 2 model: LSTM ve 1D-CNN |
| Split | SKAB: group split, BATADAL: temporal 60/20/20 |
| Senaryolar | Original, Gaussian noise, unseen-only |
| Automata parametreleri | Fixed ayar + window/alphabet sweep |
| İstatistiksel test | Wilcoxon signed-rank |
| Görseller | Confusion matrix, precision-recall curve, parameter heatmap, transition heatmap, transition graph |

Ek rapor şablonundaki cross-dataset tablo yol gösterici olarak değerlendirilmiştir. `yazlab2_v3.docx` ve proje kuralları içinde zorunlu çekirdek isterler; iki resmi veri seti, iki DL modeli, automata dönüşümü, unseen handling, noise/unseen senaryoları, parametre analizi, metrikler ve istatistiksel testlerdir. Bu repository'de iki veri seti üzerinde ayrı ayrı deney yapılmıştır; train-one-dataset/test-other-dataset şeklinde ek cross-dataset transfer deneyi yapılmamıştır ve sınırlılık olarak not edilmiştir.

## Deney Ortamı ve Süre

| Pipeline | Ortam | Yaklaşık süre |
|---|---|---:|
| Deep learning full training | Huawei MateBook 14, Intel Core Ultra 5 125H, Python 3.13.3 | 7 dakika |
| Automata full experiments | Lokal laptop ortamı | 4 dakika |

Smoke çıktıları final eğitim çıktılarından ayrı tutulmuştur. Smoke testler yalnızca kodun temel olarak çalıştığını doğrulayan hızlı kontrollerdir; final rapor sonuçları full experiment çıktılarından alınmıştır.

## Deep Learning Sonuçları

| Veri Seti | Model | Run | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|---:|
| BATADAL | 1D-CNN | 5 | 0.901 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| BATADAL | LSTM | 5 | 0.914 ± 0.022 | 0.611 ± 0.386 | 0.182 ± 0.282 | 0.227 ± 0.312 |
| SKAB | 1D-CNN | 25 | 0.881 ± 0.060 | 0.941 ± 0.071 | 0.711 ± 0.166 | 0.798 ± 0.120 |
| SKAB | LSTM | 25 | 0.908 ± 0.043 | 0.964 ± 0.051 | 0.773 ± 0.130 | 0.850 ± 0.082 |

### Deep Learning Yorumları

SKAB üzerinde LSTM en güçlü modeldir. Hem precision hem recall yüksek olduğu için F1-score da yüksektir. 1D-CNN de güçlüdür fakat LSTM daha kararlı sonuç üretmiştir.

BATADAL üzerinde accuracy değerleri yanıltıcıdır. 1D-CNN modeli neredeyse tüm örnekleri normal sınıfına attığı için accuracy yüksek görünür, ancak hiç anomali yakalayamadığı için precision, recall ve F1 sıfırdır. Bu yüzden BATADAL için F1-score ve confusion matrix özellikle önemlidir.

![SKAB LSTM confusion matrix](reports/figures/readme/skab_lstm_confusion_matrix_seed42.png)

![SKAB LSTM precision-recall curve](reports/figures/readme/skab_lstm_precision_recall_seed42.png)

![BATADAL LSTM confusion matrix](reports/figures/readme/batadal_lstm_confusion_matrix_seed42.png)

![BATADAL 1D-CNN confusion matrix](reports/figures/readme/batadal_cnn1d_confusion_matrix_seed42.png)

BATADAL 1D-CNN confusion matrix görselinde modelin anomaly sınıfını hiç tahmin etmediği görülür. Bu, accuracy'nin neden tek başına yeterli olmadığını en net gösteren çıktıdır.

## Automata Sonuçları

### Sabit Parametre ve Senaryo Sonuçları

| Veri Seti | Senaryo | Accuracy | Precision | Recall | F1-score | Unseen ratio |
|---|---|---:|---:|---:|---:|---:|
| BATADAL | original | 0.464 | 0.115 | 0.441 | 0.182 | 0.277 |
| BATADAL | gaussian_noise | 0.474 ± 0.013 | 0.117 ± 0.003 | 0.441 ± 0.000 | 0.185 ± 0.004 | 0.271 ± 0.005 |
| BATADAL | unseen_only | 0.290 | 0.135 | 0.636 | 0.222 | 0.274 |
| SKAB | original | 0.551 ± 0.006 | 0.301 ± 0.059 | 0.049 ± 0.025 | 0.077 ± 0.030 | 0.051 ± 0.013 |
| SKAB | gaussian_noise | 0.552 ± 0.007 | 0.312 ± 0.050 | 0.049 ± 0.024 | 0.078 ± 0.030 | 0.051 ± 0.013 |
| SKAB | unseen_only | 0.401 ± 0.070 | 0.196 ± 0.078 | 0.526 ± 0.149 | 0.269 ± 0.099 | 0.051 ± 0.013 |

`unseen_only` sonuçları tüm test setiyle doğrudan aynı evrende karşılaştırılmamalıdır. Bu senaryo yalnızca eğitimde görülmeyen pattern alt kümesini analiz eder. Bu nedenle original/noise sonuçlarıyla "hangisi daha iyi?" şeklinde değil, "unseen mekanizması nasıl davranıyor?" şeklinde yorumlanmalıdır.

![Automata scenario F1 comparison](reports/figures/readme/automata_scenario_f1_comparison.png)

![Automata unseen ratio comparison](reports/figures/readme/automata_scenario_unseen_ratio_comparison.png)

### Gürültü Yorumu

Gaussian noise senaryosunda F1 değerlerinde büyük düşüş oluşmamıştır:

| Veri Seti | Original F1 | Gaussian Noise F1 | Yorum |
|---|---:|---:|---|
| BATADAL | 0.182 | 0.185 ± 0.004 | Düşük seviyeli gürültü performansı bozmadı. |
| SKAB | 0.077 ± 0.030 | 0.078 ± 0.030 | Gürültü etkisi sınırlı kaldı. |

Bunun nedeni PAA ve SAX dönüşümlerinin küçük sayısal değişimleri sembolik seviyede kısmen yumuşatmasıdır. Yani küçük bir sensör gürültüsü her zaman farklı bir sembole dönüşmez.

### Parametre Analizi

Automata performansı `window_size` ve `alphabet_size` seçimlerine çok duyarlıdır.

| Veri Seti | En İyi Window | En İyi Alphabet | Accuracy | Precision | Recall | F1-score | State Count | Transition Density |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BATADAL | 6 | 5 | 0.328 | 0.154 | 0.763 | 0.257 | 229.0 | 0.0045 |
| SKAB | 6 | 6 | 0.558 ± 0.025 | 0.525 ± 0.036 | 0.382 ± 0.061 | 0.438 ± 0.030 | 125.6 | 0.0095 |

Window size arttıkça state daha uzun bağlam taşır. Örneğin `window_size=3` için state `abc` gibi kısa bir pattern iken, `window_size=6` için state daha uzun ve daha ayırt edici olur. Bu projede iki veri setinde de en iyi sonuçlar `window_size=6` ile alınmıştır.

Alphabet size arttıkça SAX daha ayrıntılı sembolik temsil üretir. Örneğin `alphabet_size=3` yalnızca `a,b,c` sembollerini kullanırken, `alphabet_size=6` daha fazla ayrım yapabilir. SKAB tarafında en iyi sonuç `alphabet_size=6` ile gelmiştir. BATADAL tarafında ise `alphabet_size=5` en iyi sonucu vermiştir; `6` değerinde state uzayı daha fazla seyrekleştiği için F1 biraz düşmüştür.

Nihai yorum:

| Veri Seti | Tercih Edilen Window | Tercih Edilen Alphabet | Gerekçe |
|---|---:|---:|---|
| SKAB | 6 | 6 | Daha uzun bağlam ve daha ayrıntılı sembolik ayrım F1'i belirgin artırdı. |
| BATADAL | 6 | 5 | Uzun bağlam yararlı oldu; ancak alphabet 6 fazla seyrek transition yapısı oluşturduğu için alphabet 5 daha dengeli kaldı. |

![Fixed vs best automata parameter F1](reports/figures/readme/automata_fixed_vs_best_parameter_f1.png)

![SKAB automata F1 heatmap](reports/figures/readme/skab_automata_parameter_f1_heatmap.png)

![BATADAL automata F1 heatmap](reports/figures/readme/batadal_automata_parameter_f1_heatmap.png)

![Automata multiseed F1 error bars](reports/figures/readme/automata_multiseed_f1_errorbars.png)

## Açıklanabilirlik

Automata modelinin en önemli farkı her karar için izlenebilir bilgi üretmesidir. Deep learning modeli genellikle sadece skor veya sınıf tahmini verir. Automata ise kararın hangi state transition üzerinden verildiğini gösterebilir.

Örnek unseen explanation çıktısı:

```json
{
  "time_step": 10,
  "previous_state": "bccc",
  "pattern": "ccca",
  "status": "unseen",
  "mapped_to": "acca",
  "probability": 1e-06,
  "prediction": 1,
  "true_label": 0,
  "edit_distance": 1,
  "path_probability_so_far": 2.116402116402116e-27,
  "decision": "anomaly",
  "confidence": 2.116402116402116e-27,
  "decision_reason": "low_probability_path"
}
```

Bu çıktının okunması:

| Alan | Anlam |
|---|---|
| `previous_state` | Modelin bir önceki sembolik durumu. |
| `pattern` | Test sırasında gelen yeni pattern. |
| `status` | Pattern eğitimde görülmüş mü, görülmemiş mi? |
| `mapped_to` | Unseen pattern için Levenshtein ile bulunan en yakın bilinen state. |
| `probability` | İlgili transition için kullanılan olasılık. |
| `edit_distance` | Gelen pattern ile eşlenen state arasındaki fark. |
| `path_probability_so_far` | O ana kadarki transition olasılıklarının birleşik değeri. |
| `decision` | Normal/anomaly kararı. |
| `confidence` | Olasılık tabanlı güven skoru. |
| `decision_reason` | Kararın ana gerekçesi. |

Transition heatmap görsellerinde satır mevcut state'i, sütun sonraki state'i gösterir. Parlak hücreler yüksek geçiş olasılığı anlamına gelir. Koyu alanların çok olması transition matrix'in seyrek olduğunu gösterir. Bu, automata modelinde beklenen bir durumdur; çünkü tüm olası sembolik geçişler eğitim verisinde görülmez.

![SKAB automata transition heatmap](reports/figures/readme/skab_automata_transition_heatmap.png)

![BATADAL automata transition heatmap](reports/figures/readme/batadal_automata_transition_heatmap.png)

Transition graph görselleri en güçlü transition ilişkilerini daha okunabilir şekilde gösterir. Bu grafikler sunumda automata modelinin "hangi state'ten hangi state'e hangi olasılıkla geçtiğini gösterebiliyor" kısmını anlatmak için kullanılabilir.

![SKAB automata transition graph](reports/figures/readme/skab_automata_transition_graph.png)

![BATADAL automata transition graph](reports/figures/readme/batadal_automata_transition_graph.png)

## İstatistiksel Analiz

Model karşılaştırmaları için Wilcoxon signed-rank testi kullanılmıştır.

| Veri Seti | Karşılaştırma | Test | Statistic | p-value | n | 0.05 seviyesinde anlamlı mı? |
|---|---|---|---:|---:|---:|---|
| SKAB | automata vs LSTM F1 | Wilcoxon signed-rank | 0.000 | 0.0625 | 5 | Hayır |
| SKAB | automata vs 1D-CNN F1 | Wilcoxon signed-rank | 0.000 | 0.0625 | 5 | Hayır |
| SKAB | LSTM vs 1D-CNN F1 | Wilcoxon signed-rank | 5.000 | 5.960e-07 | 25 | Evet |
| BATADAL | LSTM vs 1D-CNN F1 | Wilcoxon signed-rank | 0.000 | 0.1250 | 5 | Hayır |

SKAB üzerinde LSTM ve 1D-CNN farkı istatistiksel olarak anlamlıdır. Automata ile deep learning modelleri arasındaki p-value `0.0625` çıkmıştır. Bu değer `0.05` eşiğini geçmediği için "istatistiksel olarak anlamlı" raporlanmamıştır. Burada eşleşme sayısının `n=5` olması test gücünü sınırlar; bu yüzden pratik performans farkı görünse bile istatistiksel karar dikkatli yorumlanmalıdır.

McNemar testi doğrudan aynı örnekler üzerindeki iki model tahminini karşılaştırmak için uygundur. Automata tahminleri PAA/SAX sonrası pattern-transition seviyesinde, deep learning tahminleri ise sequence-window seviyesinde üretildiği için Automata vs DL karşılaştırmasında Wilcoxon signed-rank testi tercih edilmiştir.

## Genel Değerlendirme

### Deep Learning Avantajları

Deep learning modelleri çok değişkenli sensör ilişkilerini doğrudan öğrenebilir. Özellikle SKAB üzerinde LSTM ve 1D-CNN yüksek F1-score üretmiştir. Predictive performance öncelikliyse SKAB için LSTM en iyi seçimdir.

### Deep Learning Sınırlılıkları

Deep learning modelleri kararlarını doğal olarak açıklamaz. Model "anomali" dediğinde bunun hangi sensör davranışı veya hangi geçiş örüntüsü nedeniyle olduğunu doğrudan söylemek zordur. Ayrıca BATADAL örneğinde görüldüğü gibi dengesiz veri setlerinde accuracy yanıltıcı olabilir.

### Automata Avantajları

Automata modeli state, transition probability, unseen mapping ve confidence score üretebildiği için açıklanabilirlik açısından güçlüdür. Sunumda özellikle şu cümle kullanılabilir:

```text
Deep learning modeli daha yüksek skor verdi; automata modeli ise kararın hangi sembolik state geçişinden ve hangi olasılıktan kaynaklandığını gösterebildi.
```

### Automata Sınırlılıkları

Automata performansı sembolik temsil kalitesine çok bağlıdır. PCA ile tek boyuta indirgeme, PAA ile ortalama alma ve SAX ile sembolleştirme bazı ince sensör davranışlarını kaybettirebilir. Bu yüzden sabit parametrelerde performans düşük kalmıştır. Parametre sweep bu nedenle önemlidir.

## İster Kontrol Listesi

| İster | Durum | Açıklama |
|---|---|---|
| En az iki veri seti | Tamamlandı | SKAB ve BATADAL kullanıldı. |
| Resmi ham veri kullanımı | Tamamlandı | Veri setleri resmi kaynaklardan alınan ham dosyalar üzerinden hazırlandı. |
| En az iki DL modeli | Tamamlandı | LSTM ve 1D-CNN uygulandı. |
| Automata için çok boyuttan tek boyuta dönüşüm | Tamamlandı | Scaling + PCA/PC1 kullanıldı. |
| PAA/SAX sembolik temsil | Tamamlandı | Automata pipeline içinde kullanıldı. |
| Sliding-window pattern çıkarımı | Tamamlandı | Pattern'lar automata state olarak kullanıldı. |
| Probabilistic transition model | Tamamlandı | Transition probability hesaplandı. |
| Unseen pattern yönetimi | Tamamlandı | Levenshtein distance ile en yakın state mapping yapıldı. |
| Original/noise/unseen senaryoları | Tamamlandı | Automata tarafında raporlandı. |
| Window/alphabet parameter sweep | Tamamlandı | 3,4,5,6 değerleri denendi. |
| Accuracy/precision/recall/F1 | Tamamlandı | DL ve automata sonuçlarında raporlandı. |
| İstatistiksel test | Tamamlandı | Wilcoxon signed-rank testi eklendi. |
| Açıklanabilirlik çıktısı | Tamamlandı | State, transition, probability, confidence ve unseen mapping raporlandı. |

## Nasıl Çalıştırılır?

### Sanal Ortam

Bu proje yerel sanal ortam ile çalıştırılmıştır. Docker zorunlu değildir.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch CPU kurulumu gerekirse:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Bu çalışmadaki yerel Python sürümü:

```text
Python 3.13.3
```

### Testler

```powershell
python -m pytest
```

### Automata Deneyleri

```powershell
python -m src.experiments.run_full_automata_pipeline
```

### Deep Learning Deneyleri

```powershell
python -m src.experiments.run_dl_experiments
python -m src.experiments.summarize_dl_results
```

### İstatistiksel Analiz

```powershell
python -m src.experiments.run_statistical_analysis
```

Başka bir bilgisayarda çalıştırmak için repository klonlanıp aynı sanal ortam kurulabilir. Eğitim süresi CPU/GPU gücüne göre değişir. Bu çalışmada full DL training yaklaşık 7 dakika sürmüştür.

## Önemli Çıktı Dosyaları

| Tür | Dosya |
|---|---|
| DL özet tablo | `reports/tables/deep_learning/dl_summary.csv` |
| DL metrikleri | `reports/results/deep_learning/dl_evaluation_metrics.json` |
| DL training özeti | `reports/results/deep_learning/dl_training_summary.json` |
| Automata ana özet | `reports/results/automata_summary_results.csv` |
| Automata multiseed özet | `reports/results/automata_multiseed_summary.csv` |
| Automata en iyi parametre | `reports/results/automata_best_parameter_summary.csv` |
| Automata parameter sweep | `reports/results/automata_parameter_sweep_skab.csv`, `reports/results/automata_parameter_sweep_batadal.csv` |
| İstatistiksel analiz | `reports/tables/statistical_analysis_summary.csv` |
| README görselleri | `reports/figures/readme/` |
| Smoke çıktıları | `reports/results/smoke/` |

## Proje Yapısı

```text
yazlab2-timeseries-automata/
├── config.yaml
├── requirements.txt
├── README.md
├── data/
│   └── raw/
│       ├── batadal/
│       └── skab/
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── models/
│   │   ├── automata/
│   │   ├── lstm_model.py
│   │   ├── cnn1d_model.py
│   │   └── train_deep_learning.py
│   ├── evaluation/
│   ├── experiments/
│   └── visualization/
├── tests/
└── reports/
    ├── results/
    ├── figures/
    │   └── readme/
    └── tables/
```

## Sınırlılıklar ve Gelecek Çalışma

1. Automata tarafında PCA ile tek boyutlu temsil kullanıldığı için bazı sensör bilgileri kaybolabilir.
2. PAA ve SAX açıklanabilir sembolik yapı sağlasa da küçük ve kısa süreli anomalileri yumuşatabilir.
3. BATADAL veri setinde sınıf dengesizliği çok güçlüdür; bu nedenle accuracy tek başına yanıltıcıdır.
4. Unseen-only sonuçları tüm test seti sonuçlarıyla doğrudan karşılaştırılmamalıdır.
5. Train-on-one-dataset/test-on-other-dataset şeklinde ek cross-dataset transfer deneyi yapılmamıştır. Bu çalışma iki veri seti üzerinde ayrı ayrı model davranışı ve genelleme eğilimi analizi yapmıştır.
6. Automata modelinde daha gelişmiş threshold tuning, class imbalance stratejileri veya çok boyutlu sembolik temsil denenirse performans artabilir.

## Final Yorum

Bu projede deep learning modelleri özellikle SKAB üzerinde daha yüksek tahmin performansı üretmiştir. LSTM modeli SKAB için en güçlü modeldir. BATADAL üzerinde ise sınıf dengesizliği nedeniyle modeller zorlanmıştır ve 1D-CNN accuracy yüksek olmasına rağmen anomalileri yakalayamamıştır.

Automata modeli sabit parametrelerde düşük F1 üretmiştir; fakat parameter sweep ile belirgin iyileşme göstermiştir. Automata yaklaşımının asıl katkısı, karar sürecini state transition, transition probability, unseen mapping ve confidence score üzerinden açıklayabilmesidir. Bu nedenle proje, yüksek performans ile açıklanabilirlik arasında gerçek bir trade-off olduğunu göstermektedir.
