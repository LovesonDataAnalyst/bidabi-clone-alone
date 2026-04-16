#  BidaBI — Pipeline ML Reproductible avec DVC & ResNet-18

> **Classification d'images alimentaires** · Git · DVC · PyTorch · ResNet-18 · MixUp · t-SNE  
> Projet académique — Master 1 Big Data, Analyse & Business Intelligence  
> Université Panthéon-Sorbonne Paris 1 · 2025–2026

---

##  Présentation du projet

**BidaBI** est un pipeline de Machine Learning complet, reproductible et versionné, conçu pour classifier des images de produits alimentaires en plusieurs catégories. Il couvre l'ensemble du cycle de vie d'un projet ML industriel :

```
Collecte des données → Versionnement (DVC) → Entraînement (ResNet-18) → Évaluation → Release
```

Le projet met en œuvre les bonnes pratiques MLOps :
- Séparation stricte **code (Git)** / **données (DVC)**
- Pipeline structuré **RAW → INTERIM → PROCESSED**
- Environnement Python isolé et **reproductible sur toute machine**
- Releases GitHub versionnées avec archives téléchargeables

---

##  Architecture du projet

```
bidabi-clone-alone/
│
├── data/
│   ├── raw/                        # Données brutes (versionnées avec DVC)
│   │   ├── images/
│   │   │   ├── bread/              # Images de la catégorie bread
│   │   │   ├── butter/             # Images de la catégorie butter
│   │   │   ├── champagnes/         # Images de la catégorie champagnes
│   │   │   └── milk/               # Images de la catégorie milk
│   │   ├── metadata_bread_180.csv
│   │   ├── metadata_butter_180.csv
│   │   └── metadata_champagnes_180.csv
│   ├── interim/                    # Données nettoyées (à venir)
│   └── processed/                  # Données prêtes pour l'entraînement (à venir)
│
├── models/
│   ├── best_model_resnet18_finetuned.pth   # Meilleur modèle entraîné (versionné DVC)
│   └── plots/                              # Graphiques d'évaluation
│       ├── loss_curve.png
│       ├── accuracy_curve.png
│       ├── confusion_matrix.png
│       ├── per_class_accuracy.png
│       ├── roc_curves.png
│       ├── hardest_samples.png
│       └── tsne_embedding.png
│
├── src/
│   ├── asyscrapper.py              # Scraper asynchrone OpenFoodFacts
│   ├── data_loader.py              # Chargement et transformation des données
│   └── classificator.py           # Pipeline d'entraînement ResNet-18 complet
│
├── .dvc/                           # Configuration DVC
├── data/raw.dvc                    # Fichier de tracking DVC (données)
├── .gitignore                      # Exclusions Git (données, modèles, venv)
├── requirements.txt                # Dépendances Python
└── README.md
```

---

##  Installation & Reproduction

### Prérequis

- Python 3.11+
- Git
- DVC

### Étapes

**1. Cloner le dépôt**
```bash
git clone git@github.com:LovesonDataAnalyst/bidabi-clone-alone.git
cd bidabi-clone-alone
```

**2. Créer et activer l'environnement virtuel**
```bash
python -m venv .venv

# Windows
.venv\Scripts\Activate

# Linux / macOS
source .venv/bin/activate
```

**3. Installer les dépendances**
```bash
pip install -r requirements.txt
```

**4. Restaurer les données avec DVC**
```bash
dvc pull
```
> Cette commande restaure automatiquement l'intégralité du dossier `data/raw/` depuis le remote DVC local.

**5. Lancer l'entraînement**
```bash
python -m src.classificator
```

---

##  Collecte des données

Les données sont collectées via l'**API OpenFoodFacts** à l'aide d'un scraper asynchrone (`aiohttp`).

### Lancer le scraper

```bash
python src/asyscrapper.py
```

### Catégories collectées

| Catégorie    | Images | Métadonnées                        |
|--------------|--------|------------------------------------|
| `bread`      | 180    | `data/raw/metadata_bread_180.csv`  |
| `butter`     | 180    | `data/raw/metadata_butter_180.csv` |
| `champagnes` | 176    | `data/raw/metadata_champagnes_180.csv` |
| `milk`       | ~180   | `data/raw/metadata_milk_180.csv`   |

Pour changer de catégorie, modifier la variable `CATEGORY` dans `src/asyscrapper.py` :
```python
CATEGORY = "bread"  # ou "milk", "butter", "champagnes", etc.
```

---

##  Pipeline d'entraînement

### Modèle : ResNet-18 (Transfer Learning + Full Fine-Tuning)

Le pipeline implémenté dans `src/classificator.py` comprend :

| Étape | Détail |
|-------|--------|
| **Chargement** | `datasets.ImageFolder` depuis `data/raw/images/` |
| **Split** | Train 60% / Val 20% / Test 20% |
| **Augmentations** | Flip, Rotation, Affine, ColorJitter, GaussianBlur, MixUp (α=0.4) |
| **Modèle** | ResNet-18 pré-entraîné ImageNet + tête Dropout(0.4) + Linear |
| **Optimiseur** | Adam (lr=1e-5, weight_decay=1e-4) |
| **Scheduler** | CosineAnnealingLR |
| **Early Stopping** | Patience = 3 epochs |
| **Sauvegarde** | Meilleur modèle selon Val Loss → `models/best_model_resnet18_finetuned.pth` |

### Métriques d'évaluation produites

- ✅ Courbes Loss & Accuracy (Train / Validation)
- ✅ Matrice de confusion (heatmap Seaborn)
- ✅ Accuracy par classe (bar chart)
- ✅ Courbes ROC (One-vs-Rest, AUC par classe)
- ✅ Échantillons les plus difficiles (highest loss)
- ✅ Visualisation t-SNE des embeddings

---

## Versionnement des données avec DVC

Ce projet utilise **DVC** pour versionner les données et les modèles indépendamment du code.

```bash
# Ajouter une nouvelle version des données
dvc add data/raw
git add data/raw.dvc
git commit -m "Update dataset RAW"
dvc push

# Restaurer les données sur une nouvelle machine
dvc pull

# Vérifier le statut
dvc status
```

>  Le dossier `data/raw/` n'est **jamais** commité dans Git. Seul le fichier `data/raw.dvc` est versionné.

---

##  Versions du projet

| Tag | Description |
|-----|-------------|
| [`v1.0`](../../releases/tag/v1.0) | Version initiale — Sparse Checkout & nouveau dépôt Git propre |
| [`v2.0`](../../releases/tag/v2.0) | Scraper corrigé — Collecte RAW pour 3 catégories |
| [`v2.1`](../../releases/tag/v2.1) | Intégration DVC — Dataset RAW versionné et reproductible |
| [`v3.0`](../../releases/tag/v3.0) | Pipeline ML complet — ResNet-18 entraîné, évaluation complète |

---

## 🛠️ Stack technique

| Domaine | Outils |
|---------|--------|
| **Langage** | Python 3.11 |
| **Deep Learning** | PyTorch, TorchVision (ResNet-18) |
| **Data Collection** | aiohttp, OpenFoodFacts API |
| **Versionnement code** | Git, GitHub |
| **Versionnement données** | DVC |
| **Visualisation** | Matplotlib, Seaborn |
| **ML classique** | Scikit-learn (t-SNE, ROC, classification report) |
| **Environnement** | venv, pipreqs |

---

## 📊 Résultats

Les graphiques d'évaluation sont disponibles dans `models/plots/` :

| Graphique | Fichier |
|-----------|---------|
| Courbes Loss | `models/plots/loss_curve.png` |
| Courbe Accuracy | `models/plots/accuracy_curve.png` |
| Matrice de confusion | `models/plots/confusion_matrix.png` |
| Accuracy par classe | `models/plots/per_class_accuracy.png` |
| Courbes ROC | `models/plots/roc_curves.png` |
| Échantillons difficiles | `models/plots/hardest_samples.png` |
| t-SNE Embeddings | `models/plots/tsne_embedding.png` |

---

## 👤 Auteur

**Loveson SAUVEUR**  
Master 1 Big Data, Analyse & Business Intelligence — Université Panthéon-Sorbonne Paris 1  
📧 lovesonsauveur59@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/loveson-sauveur) · [GitHub](https://github.com/LovesonDataAnalyst)

---

## 📄 Licence

Ce projet est réalisé dans le cadre d'un atelier académique supervisé par le Professeur **Delnouty**.  
Dépôt source original : [`delnouty/bidabi-clone-adapt-create`](https://github.com/delnouty/bidabi-clone-adapt-create)


