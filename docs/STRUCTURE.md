# Ousia — Proje Haritası

## Ne Yapar?
Terapi seansı ses kaydı → Psikolojik kavram grafı.
Hastanın kendi sözlerinden oluşan bir ağ: hangi kavramlar birlikte geliyor, hangilerinden kaçınıyor, ses tonu ne diyor?

---

## Dizin Yapısı

```
Ouisa/
│
├── ousia/                      ← kurulabilir Python paketi
│   ├── __init__.py             ← public API (PatientGraph, SessionProcessor, ...)
│   ├── graph_engine.py         ← grafın kalbi (Hebbian, decay, LTP, silence, dissonance)
│   ├── session_processor.py    ← boru hattı yöneticisi (orchestrator)
│   ├── llm_extractor.py        ← LLM → concepts / avoidance / valence / arousal
│   ├── vocal_analyzer.py       ← openSMILE → vocal_salience (kişiye özgü baseline)
│   ├── whisper_transcriber.py  ← ses → yazı + zaman damgaları
│   └── visualizer.py           ← PatientGraph → Matplotlib görsel
│
├── tests/
│   ├── test_ousia.py           ← birim testler (graph engine, modüller)
│   └── test_realtime.py        ← streaming pipeline testi
│
├── samples/                    ← test ses dosyaları (.wav)
├── output/                     ← üretilen görseller / JSON graflar [gitignored]
│
├── pyproject.toml              ← modern Python packaging
├── requirements.txt            ← hızlı pip install için
├── .env                        ← API key'ler [gitignored, asla commit'leme]
├── .env.example                ← hangi key'lerin gerekli olduğunu gösterir
├── .gitignore
└── README.md
```

---

## Veri Akışı

```
ses dosyası (.wav)
       │
       ▼
[whisper_transcriber.py]   → yazıya döker, zaman damgası ekler
       │
       ├──────────────────────────────────────┐
       ▼                                      ▼
[vocal_analyzer.py]              [llm_extractor.py]
 openSMILE eGeMAPS               LLM (single JSON call)
 per-utterance vocal_salience    → concepts, avoidance,
 (kişiye özgü baseline)            text_valence, text_arousal
       │                                      │
       └──────────────┬───────────────────────┘
                      ▼
            dissonance = |text_arousal − vocal_salience|
                      │
                      ▼
            [graph_engine.py] → PatientGraph
             • Hebbian öğrenme
             • Kullanım-bağımlı çürüme
             • LTP eşiği
             • Sessizlik düğümü
             • Dissonans kenar özelliği
```

---

## Quickstart

```bash
# Paketi geliştirme modunda kur (import'lar her yerden çalışır)
pip install -e .

# Sadece grafı dene (ses gerektirmez):
python -m ousia.graph_engine

# Görselleştirmeyi dene:
python -m ousia.visualizer               # pencere açar
python -m ousia.visualizer output/g.png  # PNG kaydeder

# Gerçek ses dosyasıyla tam pipeline:
python -m ousia.session_processor samples/sample_speech.wav

# Testleri çalıştır:
python -m pytest tests/
```

---

## Anahtar Kavramlar

| Kavram | Ne demek? |
|--------|-----------|
| `vocal_salience` | Kişinin sesinin baseline'ından sapması (0=flat, 1=extreme) |
| `text_arousal` | LLM'in metinden çıkardığı enerji seviyesi (0=sakin, 1=heyecanlı) |
| `dissonance` | \|text_arousal − vocal_salience\| — ses ile söz çelişiyor mu? |
| `silence` node | Kaçınılan kavramlar bu düğüme kenar çizer |
| LTP | Ağırlığı 0.8'i geçen kenarlar çürümeye dirençli (travma kalıcı) |
