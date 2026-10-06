# Prédiction de souscription à un dépôt à terme — Bank Marketing

Quels clients d'une campagne de télémarketing vont souscrire à un dépôt à terme ?
Projet de classification binaire : notebook complet (déjà exécuté, résultats visibles) + interface Streamlit.

## Résultats (jeu de test, 9 043 clients jamais vus)
| | |
|---|---|
| Données | 45 211 clients, 15 variables utilisées, 11,7 % de souscripteurs |
| Modèle retenu | Random Forest (choisi sur la validation croisée parmi 7 modèles) |
| Precision / Recall / F1 | 42 % / 50 % / 0,46 (seuil de décision 0,52) |
| ROC-AUC / PR-AUC | 0,79 / 0,42 (hasard : PR-AUC ≈ 0,12) |
| Gain de ciblage | appeler 20 % des clients capte 59 % des souscripteurs |
| Variable `duration` | exclue (connue seulement après l'appel) : avec elle, ROC-AUC 0,93 ; sans, 0,80 |

## Contenu
| Fichier | Rôle |
|---|---|
| `Projet_ML_complet.ipynb` | Analyse, modélisation, évaluation (déjà exécuté) |
| `app.py` | Interface Streamlit |
| `models/` | Modèle entraîné (`pipeline.joblib`) et métadonnées (`metadata.json`) |
| `requirements.txt` | Versions des bibliothèques (le modèle en dépend) |

## Lancer l'interface
```bash
pip install -r requirements.txt
streamlit run app.py
```
Python 3.11 ou plus récent. Il n'est pas nécessaire de relancer le notebook ; pour le refaire, ajouter `bank-full.csv` (UCI, séparateur `;`) à côté et faire *Run All* (environ 25 minutes sur un seul cœur, le SVM est le plus lent).

## Choix méthodologiques
- Split stratifié avant tout traitement ; scaler, encodage et SMOTE dans un `Pipeline` (pas de fuite pendant la validation croisée).
- `GridSearchCV` optimisé sur le F1 ; modèle choisi sur la validation croisée, test utilisé une seule fois.
- Seuil de décision réglé sur des prédictions out-of-fold, puis vérifié sur le test.
- Importance des variables par permutation ; courbe de gain cumulé.

## Limites
Données d'une seule banque (2008-2010), pas de split temporel, SMOTE appliqué après One-Hot Encoding, seuil basé sur le F1 et non sur un coût en euros. Les quatre meilleurs modèles sont proches (F1 entre 0,44 et 0,46).

Données : Moro, Cortez & Rita (2014), UCI Machine Learning Repository, *Bank Marketing*.
