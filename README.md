# Winery Adventures

Pipeline di analisi dei dati dei sensori delle cisterne di fermentazione di una piccola cantina.
Legge le misurazioni di pH, temperatura e volume del mosto, aggiunge colonne derivate (medie,
conteggi, deviazione di temperatura) e calcola per ogni cisterna un **punteggio di stress da
fermentazione**. I risultati vengono salvati in un CSV e registrati su
[Weights & Biases](https://wandb.ai).

Progetto bonus del corso di Ingegneria del Software. Tecnologie: Python 3.10+, Polars, Numba, Joblib,
Weights & Biases, pytest.

## Autori

- Emanuele Unida (matricola 00284)
- Nicolo' Zucca (matricola 00269)
- Andrea Carboni (matricola 00274)

## Struttura del progetto

```
winery_adventures/
    base.py             BaseWineryAnalyzer: classe astratta con analyze_data(df) -> df
    transformations.py  WineryTransformer: colonne derivate
    computations.py     Formula di stress (Numba) e WineryHPCComputations (Joblib)
    pipeline.py         WineryPipeline: applica gli analyzer in sequenza e logga su wandb
    main.py             run_full_pipeline e riga di comando
tests/                  Test unitari e di accettazione (pytest)
data/                   Dataset di esempio (TSV)
docs/                   Documentazione Sphinx, diagrammi UML e report di performance
data_generator.py       Generatore del dataset grande
benchmark.py            Benchmark di tempi e memoria
```

## Installazione

Serve [conda](https://docs.conda.io/) (Miniconda o Anaconda). Dalla cartella del progetto:

```bash
conda env create -f environment.yml
conda activate winery
```

Per aggiornare l'ambiente dopo una modifica a `environment.yml`:

```bash
conda env update -f environment.yml --prune
```

## Uso

```bash
python -m winery_adventures.main \
    --input data/sensors_sample.tsv \
    --tank-info data/tank_info_sample.tsv \
    --output results.csv
```

| Opzione       | Descrizione                                                             |
|---------------|-------------------------------------------------------------------------|
| `--input`     | File TSV dei sensori (obbligatorio)                                     |
| `--tank-info` | File TSV con le informazioni sulle cisterne (facoltativo)               |
| `--output`    | File CSV dei risultati (default `results.csv`)                          |
| `--project`   | Nome del progetto wandb (default `WineryAdventures`)                    |

I risultati sono registrati su wandb. Prima di usarlo esegui `wandb login`, oppure imposta la variabile
d'ambiente `WANDB_MODE=offline` per non inviare nulla.

Si può usare anche da Python:

```python
from winery_adventures.main import run_full_pipeline

df = run_full_pipeline("data/sensors_sample.tsv", "data/tank_info_sample.tsv", "results.csv")
```

### Formato dei dati

- `sensors_*.tsv`: `tank_id`, `time`, `pH`, `temp`, `quantity_liters` (separati da tabulazione).
- `tank_info_*.tsv`: `tank_id`, `grape_variety` (vitigni separati da virgola), `capacity_liters`.

### Colonne aggiunte

| Colonna                        | Significato                                                                  |
|--------------------------------|------------------------------------------------------------------------------|
| `avg_pH_per_tank`              | pH medio della cisterna                                                      |
| `tank_num_readings`            | Numero di rilevazioni della cisterna                                         |
| `temperature_deviation`        | Deviazione assoluta dalla temperatura standard (26 °C)                       |
| `temperature_deviation_scaled` | Deviazione su 1000 litri (solo se è presente `quantity_liters`)              |
| `grape_variety`                | Vitigno (con le info sulle cisterne, una riga per rilevazione e vitigno)     |
| `grape_variety_num_readings`   | Numero di rilevazioni del vitigno                                            |
| `stress_score`                 | Stress da fermentazione della cisterna (formula O(n²) ottimizzata con Numba) |

## Test e qualità del codice

```bash
pytest                  # test unitari e di accettazione
ruff check .            # linting (PEP 8, import, docstring)
ruff format .           # formattazione automatica
```

Lo stesso controllo (ruff e pytest) viene eseguito dalla CI di GitHub Actions a ogni push e a ogni
pull request.

## Benchmark e performance

```bash
python data_generator.py    # crea data/full_sensors.tsv e data/full_tank_info.tsv
wandb login                 # oppure: WANDB_MODE=offline
python benchmark.py         # scrive docs/performance_report.md
```

Il report di performance confronta Python puro e Numba, e misura tempi e memoria della pipeline
sul dataset grande.

## Documentazione

```bash
sphinx-build -b html docs docs/_build/html
```

Poi apri `docs/_build/html/index.html`. La documentazione contiene il riferimento delle API
(generato dalle docstring) e i diagrammi UML (classi, sequenza, casi d'uso), che sono anche in
`docs/uml/`.

## Collaborazione

- Nessuno lavora direttamente su `main`: ogni modifica passa da un branch
  (`feature/...`, `docs/...`, `chore/...`) e da una pull request.
- Ogni pull request viene rivista da un altro membro del gruppo e deve avere la CI verde prima del merge.
- Le attività sono tracciate nelle issue del repository.
