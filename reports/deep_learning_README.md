# Deep Learning Result Interpretation

Bu bölüm deep learning deney sonuçlarının nasıl okunması gerektiğini açıklar. Aynı açıklama ana `README.md` içinde de yer almaktadır; bu dosya yalnızca deep learning artefact'lerini incelerken hızlı referans olması için eklenmiştir.

## Deep Learning Sonuç Yorumu

SKAB veri setinde LSTM ve 1D-CNN modelleri güçlü sonuç üretmiştir. LSTM `0.850 ± 0.082` F1-score ile en iyi deep learning sonucunu vermiştir. Bunun nedeni SKAB verisinin sensör zaman pencereleri içinde tekrar eden ve öğrenilebilir örüntüler içermesidir. LSTM geçmiş ölçümleri sıralı biçimde işlediği için zaman bağımlılıklarını yakalayabilmiştir. 1D-CNN de kısa lokal örüntüleri yakaladığı için güçlü kalmıştır, ancak LSTM kadar kararlı değildir.

BATADAL tarafında sonuçlar daha zordur. Veri setinde normal örnek sayısı anomali örnek sayısından çok daha fazladır. Bu dengesizlik yüzünden bir model çoğunlukla "normal" tahmini yaparak yüksek accuracy alabilir. 1D-CNN modelinin BATADAL accuracy değeri `0.901` görünmesine rağmen precision, recall ve F1-score değerleri `0.000` kalmıştır. Bu, modelin anomaly sınıfını hiç yakalayamadığı anlamına gelir. Bu nedenle BATADAL için accuracy değil, özellikle recall ve F1-score esas alınmalıdır.

BATADAL LSTM sonucu `0.227 ± 0.312` F1-score ile CNN1D'den daha iyidir, fakat standart sapması yüksektir. Bu durum modelin farklı seed'lerde çok değişken davrandığını gösterir. Yani BATADAL üzerinde LSTM bazı eğitim koşullarında anomali sinyalini yakalayabilmiş, bazı koşullarda ise zorlanmıştır.

Sunumda deep learning tarafı için çıkarılacak ana mesaj şudur:

```text
SKAB üzerinde deep learning modelleri güçlü çalıştı; BATADAL üzerinde ise sınıf dengesizliği ve veri karakteristiği modelleri zorladı. Bu yüzden accuracy tek başına yeterli değildir, F1-score ve confusion matrix birlikte yorumlanmalıdır.
```

Deep learning modelleri performans açısından özellikle SKAB'de başarılıdır; ancak kararlarını automata modeli gibi state transition ve probability üzerinden açıklayamaz. Bu nedenle proje sonucunda deep learning tarafı "yüksek tahmin performansı", automata tarafı ise "açıklanabilir karar süreci" temsil etmektedir.

## Deep Learning Training Environment and Artifact Index

Deep learning full training deneyleri Huawei MateBook 14 laptop üzerinde, Intel Core Ultra 5 125H işlemci ve Python 3.13.3 sanal ortamı ile çalıştırılmıştır. Eğitim yaklaşık 7 dakika sürmüştür. Bu süre LSTM ve 1D-CNN modellerinin SKAB ve BATADAL üzerinde 5 seed protokolüyle çalıştırılmasını kapsamaktadır.

Deep learning deneyleri için ana komutlar şunlardır:

```powershell
python -m src.experiments.run_dl_experiments
python -m src.experiments.summarize_dl_results
```

Full training çıktıları `reports/results/deep_learning/`, `reports/tables/deep_learning/` ve `reports/figures/deep_learning/` klasörlerinde tutulur. Smoke çıktıları final deneylerle karışmaması için `reports/results/smoke/` altında ayrı tutulmuştur. Smoke testlerin amacı model kalitesini ölçmek değil, kodun hızlı biçimde çalışabildiğini doğrulamaktır.

Rapor için seçilen deep learning görselleri `reports/figures/readme/` klasörüne kopyalanmıştır. Bu klasördeki dosyalar, ham deney çıktılarının düzenlenmiş ve README içinde kullanılan kopyalarıdır:

| Dosya | Ne gösteriyor? |
|---|---|
| `skab_lstm_confusion_matrix_seed42.png` | SKAB üzerinde LSTM modelinin doğru/yanlış sınıflandırma dağılımı. |
| `skab_lstm_precision_recall_seed42.png` | SKAB üzerinde LSTM modelinin precision-recall davranışı. |
| `batadal_lstm_confusion_matrix_seed42.png` | BATADAL üzerinde LSTM modelinin anomaly yakalama davranışı. |
| `batadal_cnn1d_confusion_matrix_seed42.png` | BATADAL üzerinde CNN1D modelinin anomaly sınıfını yakalayamadığını gösteren confusion matrix. |

Bu artefact seçimi özellikle BATADAL'daki accuracy/F1 çelişkisini ve SKAB'deki LSTM başarısını sunumda hızlı anlatmak için yapılmıştır.
