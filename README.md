# Project 2 – DistilBERT Hyperparameter Tuning (Containerized)

Dieses Repository enthält die in Python-Skripte überführte Version des ursprünglichen Notebooks `src/mlops_hyperparameter_tuning.ipynb` sowie alle Artefakte für die Containerisierung (Tasks 1–4).

## 1. Lokale Entwicklung

### Installation (uv)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # einmalig (oder via Paketmanager)
uv venv
source .venv/bin/activate
uv pip install --editable .
```

### Einzelnes Training (Task 1)
Der einfachste Einstieg ist `main.py`, der alle Hyperparameter und den Speicherpfad als Argumente akzeptiert:
```bash
python main.py \
  --run-name local_best \
  --learning-rate 3e-5 \
  --weight-decay 0.0001 \
  --train-batch-size 64 \
  --warmup-steps 0 \
  --checkpoint-dir models/local_best \
  --wandb-login
```
* Der Lauf wird in Weights & Biases geloggt (setze `WANDB_API_KEY` oder übergib `--wandb-api-key`).
* Checkpoints landen im angegebenen Verzeichnis (Lightning `ModelCheckpoint`).
* Standardmässig werden Runs im W&B-Projekt `mlops_project2` gespeichert (änderbar via `--project-name`).

### Erweiterte Workflows
- **CLI-Hilfe:** `python -m mlops_hyperparameter_tuning.cli --help`
- **Vorbereiteter Experimentplan:** `python -m mlops_hyperparameter_tuning.cli run-plan --wandb-login`
- **Custom Sweep:** `python -m mlops_hyperparameter_tuning.cli run-sweep --count 8 --wandb-login`

## 2. Docker (Tasks 2 & 3)

Der Container läuft standardmässig mit den neuen Parametern (`lr=3e-5`, `wd=0.0001`, `batch=64`, `warmup=0`) und speichert Checkpoints in `/artifacts`.
Alle Runs landen dabei ebenfalls im W&B-Projekt `mlops_project2`, sofern nicht anders angegeben.

```bash
# Image bauen
docker build -t mlops-hpo .

# Training lokal starten (WANDB in den Container durchreichen)
docker run --rm \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -v "$(pwd)/artifacts:/artifacts" \
  mlops-hpo 
```

Andere Hyperparameter lassen sich durch zusätzliche Argumente überschreiben, z. B.:
```bash
docker run --rm mlops-hpo \
  --run-name sweep_candidate \
  --learning-rate 3e-5 \
  --checkpoint-dir /artifacts/sweep_candidate
```

### Codespaces / Docker Playground (Task 3)
1. Repository klonen.
2. `docker build -t mlops-hpo .`
3. `docker run --rm mlops-hpo` (gegebenenfalls `WANDB_API_KEY` setzen).
4. Ergebnisse in W&B prüfen und mit lokalem Lauf vergleichen.


## 3. Struktur

```
├── main.py                         # Single-run entrypoint (Task 1 requirement)
├── pyproject.toml                  # Dependency metadata (uv compatible)
├── Dockerfile / .dockerignore      # Container build + context
├── README.md                       # Bedienungsanleitung
├── report.md                       # Vorlage für den einzureichenden Bericht
└── src/mlops_hyperparameter_tuning
    ├── data.py / model.py          # Lightning DataModule + Module
    ├── experiments.py / sweeps.py  # Lauf- & Sweep-Logik
    ├── cli.py                      # Mehrstufige Workflows
    └── mlops_hyperparameter_tuning.ipynb (Original)
```
