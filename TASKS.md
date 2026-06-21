# TASKS.md

# Optimisation intelligente d’un portefeuille d’investissement sous contraintes

## Objectif du projet

Développer une application intelligente d’aide à la décision pour l’optimisation d’un portefeuille d’investissement basé sur les actions cotées à la Bourse de Casablanca.

Le pipeline global du projet est :

```text
Données → Facteurs financiers → Scoring → Optimisation → Simulation → Recommandation
```

L’application finale doit permettre à l’utilisateur de saisir son budget, son horizon d’investissement, son profil de risque et son objectif financier, puis afficher les scores des actions, l’allocation optimale, les montants à investir et les résultats de simulation.

---

# Phase 0 — Initialisation du projet

## 0.1 Créer la structure du projet

* [ ] Créer le dépôt GitHub du projet.
* [ ] Créer l’environnement virtuel Python.
* [ ] Créer les dossiers principaux :

```text
app/
src/
data/raw/
data/processed/
data/outputs/
models/
notebooks/
tests/
```

* [ ] Ajouter les fichiers initiaux :

```text
README.md
AGENTS.md
TASKS.md
requirements.txt
.gitignore
```

## 0.2 Installer les dépendances

* [ ] Ajouter les bibliothèques nécessaires dans `requirements.txt` :

```text
pandas
numpy
scikit-learn
scipy
streamlit
plotly
matplotlib
joblib
pytest
openpyxl
```

* [ ] Installer les dépendances :

```bash
pip install -r requirements.txt
```

## 0.3 Vérifier la structure

* [ ] Vérifier que le projet peut être ouvert dans VS Code.
* [ ] Vérifier que l’environnement Python est bien activé.
* [ ] Faire un premier commit Git :

```bash
git add .
git commit -m "Initial project structure"
```

---

# Phase 1 — Collecte des données

## Objectif

Récupérer les historiques de prix et de volumes des actions cotées à la Bourse de Casablanca.

## Fichier à créer

```text
src/data_collection.py
```

## Tâches

* [ ] Créer une fonction `collect_stock_data(tickers, start_date=None, end_date=None)`.
* [ ] Utiliser BVCscrap si disponible.
* [ ] Retourner un DataFrame pandas contenant au minimum :

```text
date
ticker
close
low
high
volume
variation
```

* [ ] Gérer les erreurs si BVCscrap n’est pas installé.
* [ ] Prévoir une fonction de sauvegarde des données brutes dans `data/raw/`.
* [ ] Tester la collecte sur quelques actions.
* [ ] Vérifier que les données récupérées contiennent bien les prix et les volumes.

## Résultat attendu

```text
data/raw/raw_prices.csv
```

---

# Phase 2 — Nettoyage des données

## Objectif

Nettoyer, harmoniser et préparer les données financières avant le calcul des rendements et des indicateurs.

## Fichier à créer

```text
src/data_cleaning.py
```

## Tâches

* [ ] Créer une fonction `clean_prices(df)`.
* [ ] Harmoniser les colonnes :

```text
Value → close
Min → low
Max → high
Volume → volume
Variation → variation
```

* [ ] Convertir la colonne `date` au format datetime.
* [ ] Convertir les colonnes de prix et de volume en valeurs numériques.
* [ ] Supprimer les lignes sans prix disponible.
* [ ] Supprimer les doublons.
* [ ] Supprimer les lignes où `close <= 0`.
* [ ] Trier les données par `ticker` et par `date`.
* [ ] Créer une fonction `save_clean_prices(df, output_path)`.
* [ ] Sauvegarder les données nettoyées dans `data/processed/`.

## Résultat attendu

```text
data/processed/clean_prices.csv
```

---

# Phase 3 — Calcul des rendements

## Objectif

Calculer les rendements journaliers des actions à partir des prix nettoyés.

## Fichier à modifier

```text
src/features.py
```

## Tâches

* [ ] Créer une fonction `calculate_daily_returns(clean_prices_df)`.
* [ ] Utiliser la colonne `close`.
* [ ] Calculer le rendement journalier avec `pct_change()`.
* [ ] Retourner un DataFrame avec :

  * dates en index ;
  * tickers en colonnes ;
  * rendements en valeurs.
* [ ] Supprimer les lignes où tous les rendements sont manquants.
* [ ] Sauvegarder les rendements dans `data/processed/`.

## Résultat attendu

```text
data/processed/returns.csv
```

---

# Phase 4 — Construction des facteurs financiers

## Objectif

Construire les variables financières qui seront utilisées par le modèle de scoring.

## Fichier à modifier

```text
src/features.py
```

## Facteurs à calculer

* [ ] Rendement passé sur 20 jours.
* [ ] Momentum sur 20 jours.
* [ ] Volatilité sur 20 jours.
* [ ] RSI sur 14 jours.
* [ ] MACD.
* [ ] Signal MACD.
* [ ] Moyenne mobile 20 jours.
* [ ] Moyenne mobile 50 jours.
* [ ] Volume moyen 20 jours.
* [ ] Drawdown.

## Tâches

* [ ] Créer une fonction `build_financial_features(clean_prices_df)`.
* [ ] Calculer les facteurs pour chaque action.
* [ ] Respecter l’ordre chronologique.
* [ ] Éviter tout look-ahead bias.
* [ ] Ne jamais utiliser de données futures dans les features.
* [ ] Retourner un DataFrame long format :

```text
date
ticker
close
past_return_20d
momentum_20d
volatility_20d
rsi_14d
macd
macd_signal
moving_average_20d
moving_average_50d
average_volume_20d
drawdown
```

* [ ] Supprimer ou traiter les lignes avec valeurs manquantes.
* [ ] Sauvegarder les features dans `data/processed/`.

## Résultat attendu

```text
data/processed/features.csv
```

---

# Phase 5 — Construction de la variable cible

## Objectif

Créer la variable cible utilisée pour entraîner le modèle de scoring.

## Fichier à modifier

```text
src/features.py
```

## Méthode

La cible vaut 1 si le rendement futur sur 20 jours est supérieur à 2%.

```text
target = 1 si future_return_20d > 0.02
target = 0 sinon
```

## Tâches

* [ ] Créer une fonction `add_target_variable(features_df, horizon_days=20, threshold=0.02)`.
* [ ] Calculer le rendement futur sur 20 jours :

```text
future_return = close.shift(-20) / close - 1
```

* [ ] Créer la colonne `target`.
* [ ] Supprimer les lignes où `future_return` est manquant.
* [ ] Vérifier que `future_return` n’est pas utilisé comme variable explicative.
* [ ] Sauvegarder le dataset final du scoring.

## Résultat attendu

```text
data/processed/scoring_dataset.csv
```

---

# Phase 6 — Modèle de scoring des actions

## Objectif

Entraîner une régression logistique pour attribuer à chaque action un score d’attractivité entre 0 et 1.

## Fichier à créer

```text
src/scoring.py
```

## Tâches

* [ ] Créer une fonction `get_feature_columns()`.
* [ ] Créer une fonction `train_scoring_model(scoring_dataset, test_size=0.2)`.
* [ ] Utiliser une séparation chronologique des données.
* [ ] Ne pas utiliser de split aléatoire.
* [ ] Standardiser les variables avec `StandardScaler`.
* [ ] Entraîner un modèle `LogisticRegression`.
* [ ] Calculer les métriques :

```text
accuracy
precision
recall
f1_score
roc_auc
confusion_matrix
```

* [ ] Sauvegarder le modèle dans `models/logistic_model.pkl`.
* [ ] Sauvegarder le scaler dans `models/scaler.pkl`.
* [ ] Créer une fonction `score_latest_stocks(features_df, model, scaler)`.
* [ ] Générer les scores récents des actions.
* [ ] Sauvegarder les scores dans `data/outputs/`.

## Résultat attendu

```text
models/logistic_model.pkl
models/scaler.pkl
data/outputs/latest_scores.csv
```

---

# Phase 7 — Optimisation du portefeuille

## Objectif

Transformer les scores d’attractivité en allocation optimale sous contraintes.

## Fichier à créer

```text
src/optimization.py
```

## Méthode

Maximiser :

```text
score_portefeuille - lambda_risk * risque_portefeuille
```

Avec :

```text
score_portefeuille = w.T @ scores
risque_portefeuille = w.T @ covariance_matrix @ w
```

## Contraintes

* [ ] Somme des poids = 1.
* [ ] Poids positifs.
* [ ] Pas de vente à découvert.
* [ ] Poids maximum par action.
* [ ] Prise en compte du profil de risque.

## Profils de risque

```text
conservative:
  lambda_risk = 10
  max_weight = 0.20

balanced:
  lambda_risk = 5
  max_weight = 0.30

dynamic:
  lambda_risk = 2
  max_weight = 0.40
```

## Tâches

* [ ] Créer une fonction `optimize_portfolio(scores_df, returns_df, risk_profile="balanced")`.
* [ ] Utiliser `scipy.optimize.minimize`.
* [ ] Utiliser la méthode SLSQP.
* [ ] Aligner les tickers présents dans `scores_df` et `returns_df`.
* [ ] Calculer la matrice de covariance des rendements.
* [ ] Retourner un DataFrame contenant :

```text
ticker
weight
score
```

* [ ] Vérifier que la somme des poids est égale à 1.
* [ ] Vérifier qu’aucun poids n’est négatif.
* [ ] Sauvegarder l’allocation dans `data/outputs/`.

## Résultat attendu

```text
data/outputs/optimal_allocation.csv
```

---

# Phase 8 — Calcul des montants à investir

## Objectif

Convertir les poids du portefeuille en montants concrets selon le budget de l’utilisateur.

## Fichier à modifier

```text
src/optimization.py
```

## Tâches

* [ ] Créer une fonction `compute_investment_amounts(allocation_df, budget)`.
* [ ] Vérifier que le budget est positif.
* [ ] Ajouter une colonne `amount`.
* [ ] Ajouter une colonne `amount_rounded`.
* [ ] Sauvegarder les montants à investir.

## Résultat attendu

```text
data/outputs/investment_amounts.csv
```

---

# Phase 9 — Simulation Monte Carlo par bootstrap historique

## Objectif

Évaluer plusieurs scénarios possibles de l’allocation recommandée à partir des rendements historiques observés.

## Fichier à créer

```text
src/simulation.py
```

## Méthode

* Utiliser les rendements historiques du portefeuille.
* Tirer aléatoirement des rendements avec remplacement.
* Simuler l’évolution future de la valeur du portefeuille.
* Répéter l’opération plusieurs milliers de fois.

## Tâches

* [ ] Créer une fonction `historical_bootstrap_monte_carlo(...)`.
* [ ] Aligner les tickers entre les rendements et l’allocation.
* [ ] Calculer les rendements historiques du portefeuille.
* [ ] Simuler `n_simulations` trajectoires.
* [ ] Utiliser un horizon en jours.
* [ ] Calculer les indicateurs :

```text
mean_final_value
median_final_value
percentile_5
percentile_95
max_loss_simulated
probability_target_reached
```

* [ ] Retourner les trajectoires simulées.
* [ ] Retourner un résumé statistique.
* [ ] Sauvegarder les résultats dans `data/outputs/`.

## Résultats attendus

```text
data/outputs/simulation_paths.csv
data/outputs/simulation_summary.csv
```

---

#

# Phase 10 — Application Streamlit

## Objectif

Créer une interface interactive permettant à l’utilisateur d’utiliser le système sans manipuler le code.

## Fichier à créer

```text
app/streamlit_app.py
```

## Entrées utilisateur

* [ ] Budget en DH.
* [ ] Horizon d’investissement.
* [ ] Profil de risque :

```text
conservative
balanced
dynamic
```

* [ ] Objectif financier ou rendement cible.

## Affichages attendus

* [ ] Titre et description du projet.
* [ ] Tableau des scores des actions.
* [ ] Allocation recommandée.
* [ ] Montants à investir.
* [ ] Graphique de répartition du portefeuille.
* [ ] Risque du portefeuille.
* [ ] Résumé de la simulation Monte Carlo.
* [ ] Graphique des scénarios simulés.
* [ ] Probabilité d’atteindre l’objectif.
* [ ] Recommandation finale.

## Tâches

* [ ] Charger les données nettoyées.
* [ ] Charger le modèle et le scaler.
* [ ] Générer les scores.
* [ ] Optimiser le portefeuille selon les paramètres utilisateur.
* [ ] Lancer la simulation.
* [ ] Afficher les résultats clairement.
* [ ] Ajouter des messages d’erreur si les fichiers sont manquants.
* [ ] Tester l’application localement.

## Commande de lancement

```bash
streamlit run app/streamlit_app.py
```

---

# Phase 11 — Tests unitaires

## Objectif

Vérifier que les fonctions principales fonctionnent correctement.

## Dossier à utiliser

```text
tests/
```

## Tests à créer

* [ ] `test_data_cleaning.py`
* [ ] `test_features.py`
* [ ] `test_scoring.py`
* [ ] `test_optimization.py`
* [ ] `test_simulation.py`

## Points à vérifier

* [ ] Les données nettoyées ne contiennent pas de prix négatifs.
* [ ] Les rendements sont correctement calculés.
* [ ] Les features ne contiennent pas de données futures.
* [ ] La variable cible est correctement construite.
* [ ] Les poids optimisés sont positifs.
* [ ] La somme des poids vaut 1.
* [ ] La simulation retourne le bon nombre de trajectoires.
* [ ] Les métriques de simulation sont bien calculées.

## Commande

```bash
pytest
```

---

# Phase 12 — Documentation

## Objectif

Documenter clairement le projet pour le rapport, la soutenance et GitHub.

## Fichier à compléter

```text
README.md
```

## Contenu attendu

* [ ] Contexte du projet.
* [ ] Problématique.
* [ ] Objectifs.
* [ ] Pipeline global.
* [ ] Structure du projet.
* [ ] Installation.
* [ ] Exécution.
* [ ] Description du module de données.
* [ ] Description du module de scoring.
* [ ] Description du module d’optimisation.
* [ ] Description du module de simulation.
* [ ] Description de l’application Streamlit.
* [ ] Limites.
* [ ] Perspectives.

---

# Phase 13 — Validation finale

## Objectif

Vérifier que le projet est complet et cohérent avant présentation.

## Checklist finale

* [ ] Le projet s’installe sans erreur.
* [ ] Les données sont correctement chargées.
* [ ] Les rendements sont calculés.
* [ ] Les facteurs financiers sont générés.
* [ ] Le modèle de scoring s’entraîne correctement.
* [ ] Les scores sont compris entre 0 et 1.
* [ ] L’optimisation respecte les contraintes.
* [ ] La somme des poids vaut 1.
* [ ] Les montants à investir sont cohérents avec le budget.
* [ ] La simulation Monte Carlo fonctionne.
* [ ] L’application Streamlit se lance.
* [ ] Les résultats sont compréhensibles.
* [ ] Les tests passent.
* [ ] Le README est complet.
* [ ] Les captures d’écran sont prêtes pour le rapport.
* [ ] Les limites du projet sont clairement expliquées.

#

---

# Ordre recommandé de développement avec Codex

1. Structure du projet.
2. Collecte des données.
3. Nettoyage des données.
4. Calcul des rendements.
5. Construction des facteurs financiers.
6. Construction de la variable cible.
7. Modèle de scoring.
8. Optimisation du portefeuille.
9. Calcul des montants à investir.
10. Simulation Monte Carlo bootstrap.
11. Application Streamlit.
12. Tests unitaires.
13. Backtesting.
14. Documentation.
15. Validation finale.

---

# Règle importante pour Codex

Ne jamais développer tout le projet en une seule fois.

Chaque tâche doit être traitée séparément, testée, validée, puis commitée avant de passer à la suivante.

Workflow recommandé :

```bash
git checkout -b feature/nom-du-module
```

Après chaque module :

bash
pytest
git add .
git commit -m "Add nom du module"

Puis demander une revue à Codex :

Review the last changes. Check for bugs, financial methodology errors, look-ahead bias, path issues, missing tests and code quality problems.
