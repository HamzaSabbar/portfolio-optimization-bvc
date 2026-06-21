# Optimisation intelligente d’un portefeuille d’investissement sous contraintes

## Description du projet

Ce projet porte sur la conception et le développement d’une application intelligente d’aide à la décision pour l’optimisation d’un portefeuille d’investissement. Il s’intéresse aux actions cotées à la Bourse de Casablanca et vise à proposer une allocation adaptée au budget, au profil de risque et à l’objectif financier de l’utilisateur.

La solution développée repose sur une chaîne complète allant de la collecte des données financières jusqu’à la recommandation finale du portefeuille. Elle combine plusieurs étapes : préparation des données, calcul de facteurs financiers, scoring des actions, optimisation sous contraintes, simulation Monte Carlo par bootstrap historique et visualisation interactive avec Streamlit.

---

## Objectifs

Les principaux objectifs du projet sont :

* collecter les données historiques des actions cotées à la Bourse de Casablanca ;
* nettoyer et préparer les données financières ;
* calculer les rendements journaliers et plusieurs indicateurs financiers ;
* construire un modèle de scoring basé sur une régression logistique ;
* attribuer à chaque action un score d’attractivité compris entre 0 et 1 ;
* optimiser la répartition du capital selon le score, le risque et les contraintes ;
* simuler plusieurs scénarios futurs possibles du portefeuille ;
* développer une application interactive permettant de visualiser les résultats.

---

## Pipeline général

Le fonctionnement global du projet suit la chaîne suivante :

```text
Données historiques
        ↓
Nettoyage et préparation
        ↓
Calcul des rendements
        ↓
Construction des facteurs financiers
        ↓
Scoring des actions
        ↓
Optimisation du portefeuille
        ↓
Simulation Monte Carlo
        ↓
Application Streamlit
```

---

## Méthodologie

### 1. Collecte des données

Les données historiques des actions sont collectées à partir de la Bourse de Casablanca. Elles contiennent notamment les prix, les volumes et les variations des actions étudiées.

### 2. Préparation des données

Les données brutes sont nettoyées et harmonisées afin d’obtenir une base exploitable. Cette étape comprend notamment :

* la conversion des dates ;
* l’harmonisation des noms de colonnes ;
* la conversion des prix et volumes en valeurs numériques ;
* la suppression des valeurs invalides ;
* le tri chronologique par action.

### 3. Calcul des facteurs financiers

Plusieurs facteurs financiers sont calculés afin de décrire chaque action selon différentes dimensions : performance, tendance, risque et liquidité.

Les principaux facteurs utilisés sont :

* rendement passé ;
* momentum ;
* volatilité ;
* RSI ;
* MACD ;
* moyennes mobiles ;
* volume moyen ;
* drawdown.

### 4. Scoring des actions

Un modèle de régression logistique est utilisé pour attribuer un score d’attractivité à chaque action. La variable cible est construite à partir du rendement futur sur un horizon de 20 jours.

Le score obtenu est compris entre 0 et 1. Il permet de classer les actions selon leur attractivité relative.

### 5. Optimisation du portefeuille

Les scores des actions sont ensuite utilisés dans un module d’optimisation. L’objectif est de maximiser le score global du portefeuille tout en pénalisant le risque.

L’optimisation respecte plusieurs contraintes :

* somme des poids égale à 1 ;
* interdiction de la vente à découvert ;
* poids positifs ;
* poids maximum par action ;
* prise en compte du profil de risque de l’utilisateur.

### 6. Simulation Monte Carlo par bootstrap historique

Une simulation Monte Carlo par bootstrap historique est utilisée pour évaluer plusieurs scénarios futurs possibles. Cette méthode rééchantillonne les rendements historiques du portefeuille optimisé afin de simuler son évolution possible.

Les indicateurs simulés incluent notamment :

* valeur finale moyenne ;
* valeur finale médiane ;
* percentile 5 % ;
* percentile 95 % ;
* perte maximale simulée ;
* probabilité d’atteindre l’objectif financier.

---

## Structure du projet

```text
portfolio-optimization-bvc/
│
├── app/
│   ├── streamlit_app.py
│   └── data_preview.py
│
├── src/
│   ├── data_collection.py
│   ├── data_cleaning.py
│   ├── features.py
│   ├── scoring.py
│   ├── optimization.py
│   ├── simulation.py
│   └── utils.py
│
├── scripts/
│   ├── run_pipeline.py
│   ├── validate_raw_data.py
│   ├── diagnose_data_quality.py
│   └── diagnose_results.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── outputs/
│
├── models/
│   ├── logistic_model.pkl
│   └── scaler.pkl
│
├── tests/
│
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Cloner le projet

```bash
git clone <url-du-repository>
cd portfolio-optimization-bvc
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
```

Sous Windows :

```bash
.venv\Scripts\activate
```

Sous Linux ou macOS :

```bash
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
python -m pip install -r requirements.txt
```

---

## Utilisation

### 1. Collecter les données

Pour collecter les données de toutes les entreprises disponibles :

```bash
python src\data_collection.py --all --start-date 2022-01-01 --end-date 2024-12-31 --output data\raw\raw_prices.csv
```

Si une liste personnalisée de tickers est utilisée :

```bash
python src\data_collection.py --tickers-file config\bvc_tickers.txt --start-date 2022-01-01 --end-date 2024-12-31 --output data\raw\raw_prices.csv
```

### 2. Vérifier les données brutes

```bash
python scripts\validate_raw_data.py
```

### 3. Prévisualiser les données

```bash
streamlit run app\data_preview.py
```

Cette interface permet de visualiser les données brutes collectées : nombre d’entreprises, nombre de lignes par entreprise, plages de dates, valeurs manquantes, prix et volumes.

### 4. Lancer le pipeline complet

```bash
python scripts\run_pipeline.py
```

Ce script génère les fichiers nécessaires à l’application finale :

```text
data/processed/clean_prices.csv
data/processed/returns.csv
data/processed/features.csv
data/processed/scoring_dataset.csv
models/logistic_model.pkl
models/scaler.pkl
data/outputs/latest_scores.csv
```

### 5. Lancer l’application finale

```bash
streamlit run app\streamlit_app.py
```

---

## Application Streamlit

L’application finale permet à l’utilisateur de saisir :

* son budget ;
* son horizon d’investissement ;
* son profil de risque ;
* son objectif financier ;
* le nombre de simulations Monte Carlo.

Elle affiche ensuite :

* les scores des actions ;
* l’allocation optimale ;
* les montants à investir ;
* les indicateurs de risque ;
* les résultats de la simulation Monte Carlo ;
* une recommandation finale interprétable.

---

## Profils de risque

Trois profils de risque sont proposés :

| Profil       | Description                                                                         |
| ------------ | ----------------------------------------------------------------------------------- |
| Conservateur | Favorise une allocation plus prudente avec une pénalisation plus forte du risque.   |
| Équilibré    | Cherche un compromis entre score d’attractivité et risque du portefeuille.          |
| Dynamique    | Accepte un niveau de risque plus élevé afin de favoriser les actions mieux scorées. |

---

## Résultats générés

Les principaux résultats du projet sont stockés dans le dossier `data/outputs/`.

Exemples :

```text
latest_scores.csv
model_metrics.csv
model_coefficients.csv
collection_report.csv
excluded_tickers_report.csv
data_quality_report.csv
```

Ces fichiers permettent d’analyser les scores obtenus, les performances du modèle, les actions exclues, la qualité des données et les diagnostics finaux.

---

## Tests

Pour exécuter les tests unitaires :

```bash
pytest
```

Les tests vérifient notamment :

* le nettoyage des données ;
* le calcul des rendements ;
* la construction des facteurs financiers ;
* la construction de la variable cible ;
* le scoring ;
* l’optimisation ;
* la simulation Monte Carlo.

---

## Limites du projet

Ce projet présente certaines limites :

* les résultats dépendent fortement de la qualité des données historiques ;
* les performances passées ne garantissent pas les performances futures ;
* le modèle de scoring reste basé sur des facteurs financiers historiques ;
* la simulation Monte Carlo illustre des scénarios possibles, mais ne constitue pas une prévision certaine ;
* les frais de transaction, la fiscalité et certaines contraintes de liquidité ne sont pas intégrés de manière complète.

---

## Perspectives d’amélioration

Plusieurs améliorations peuvent être envisagées :

* intégrer des données fondamentales des entreprises ;
* ajouter les frais de transaction et la fiscalité ;
* intégrer un benchmark comme le MASI ;
* tester d’autres modèles de machine learning ;
* améliorer le backtesting ;
* ajouter une base de données ;
* déployer l’application en ligne ;
* permettre l’export automatique des résultats en PDF ou Excel.

---

## Avertissement

Cette application est développée dans un cadre académique. Elle constitue un outil d’aide à la décision et ne doit pas être considérée comme un conseil financier. Les décisions d’investissement comportent des risques et doivent être prises avec prudence.
