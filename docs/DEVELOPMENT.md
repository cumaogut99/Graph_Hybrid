# Geliştirme Notları

## Klasör yapısı

```
app.py                     Giriş noktası (PyInstaller bunu paketler): başlangıç ayarları ve TimeGraphApp
src/
  operations/              TimeGraphApp işlemleri: dosya açma, proje, layout, veri kaydetme
  data/                    Okuma, dönüştürme (CSV/Excel/TDMS -> MPAI), SignalProcessor
    signal_processing/     SignalProcessor parçaları (views, filtering, normalization, statistics, queries)
    import_dialog/         DataImportDialog parçaları (layout, preview, theme, önizleme thread'i)
    csv_conversion/        CsvToMpaiConverter parçaları (reading, time_column, numeric, writing, zaman kalıpları)
  managers/                Panel ve davranış yöneticileri
    plot/                  PlotManager parçaları (lod, subplots, signals, view, markers) ve yardımcıları
    correlations/          CorrelationsPanelManager parçaları (panel_ui, calculation, parameter_dialog)
  ui/                      Pencere, panel ve dialog'lar
    main_window/           TimeGraphApp parçaları (loading, saving, files, shutdown)
    time_graph_widget.py   Her dosyanın ana widget'ı
    time_graph/            TimeGraphWidget parçaları (tabs, cursor_statistics, graph_settings, filters, panels, theme, layout_config)
    statistics/            StatisticsPanel parçaları (columns, graph_sections, signal_rows, values) ve satır widget'ları
  graphics/, reporting/, utils/
tests/                     pytest testleri
tools/refactor/            Büyük sınıfları davranış değiştirmeden bölme araçları
```

Büyük sınıflar **mixin**'lere bölünmüştür: ana sınıf (`TimeGraphApp`, `SignalProcessor`,
`TimeGraphWidget`, `PlotManager`, `StatisticsPanel`, `DataImportDialog`,
`CsvToMpaiConverter`, `CorrelationsPanelManager`)
`__init__`'i, Qt sinyallerini ve çekirdek akışı tutar; diğer metotlar alt
klasördeki mixin'lerde durur. Mixin'ler düz sınıflardır (QObject'ten türemez),
durumu ana sınıfın `__init__`'i kurar, `self` aynıdır. Yeni bir metot ilgili
mixin'e eklenir; import yolları (`from src.data.signal_processor import SignalProcessor`) değişmez.

## Testler

```
pip install -r requirements-dev.txt
pytest
```

Testler Qt'yi ekransız çalıştırır, geçici klasörde çalışır ve gerçek
`%LOCALAPPDATA%` önbelleğine dokunmaz. `tests/test_smoke_app.py` ana pencereyi
açar, bir CSV'yi import dialog'u ve yükleme thread'i üzerinden yükler, kanal
çizer, normalizasyon, sekme ve layout akışlarını çalıştırır.

## Büyük bir dosyayı düşük riskle bölmek

Kural: bir commit'te yalnızca taşıma; isim veya mantık değişikliği ayrı commit'te.

1. Önce davranışı kilitleyen testler (yoksa) ve bir snapshot:
   `python tools/refactor/snapshot.py before.json src.x.module:Class`
2. Plan dosyası yaz (örnek: `tools/refactor/example_plan.json`) ve böl:
   `python tools/refactor/split_class.py plan.json`
   Metotlar ve üstlerindeki yorumlar aynen taşınır, mixin'e orijinal modülün
   importları kopyalanıp kullanılmayanlar silinir, tanımsız kalan isimler raporlanır.
   Silinecek ölü metotlar için: `python tools/refactor/remove_methods.py dosya.py Sınıf metot ...`
3. Ana dosyada artık kullanılmayan importları temizle:
   `python tools/refactor/prune_new_unused.py eski_kopya.py src/x/module.py`
4. Doğrula: her metodun sözdizim ağacı aynı mı, hiçbir yorum kayboldu mu, testler geçiyor mu?
   ```
   python tools/refactor/snapshot.py after.json src.x.module:Class
   python tools/refactor/compare.py before.json after.json Class
   python tools/refactor/comments.py eski_kopya.py -- src/x/module.py src/x/parts/*.py
   pytest
   ```

Aynı isimle iki kez tanımlanmış metotlar (Python'da sonuncusu geçerlidir)
aynı mixin'e, aynı sırayla taşınmalıdır; `compare.py` bunu yakalar.
