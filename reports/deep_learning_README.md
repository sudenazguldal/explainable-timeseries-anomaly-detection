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
