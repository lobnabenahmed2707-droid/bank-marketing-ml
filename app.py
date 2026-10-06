"""Interface Streamlit — Prédiction de souscription à un dépôt à terme (Bank Marketing).

Lancer :  streamlit run app.py
Prérequis : avoir exécuté le notebook, qui crée models/pipeline.joblib et models/metadata.json.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import streamlit as st

BASE = Path(__file__).parent / "models"

st.set_page_config(page_title="Ciblage dépôt à terme", page_icon="📞", layout="wide")


@st.cache_resource
def load_artifacts():
    pipe = joblib.load(BASE / "pipeline.joblib")
    with open(BASE / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    return pipe, meta


if not (BASE / "pipeline.joblib").exists() or not (BASE / "metadata.json").exists():
    st.title("Ciblage dépôt à terme")
    st.error("Modèle introuvable. Exécute d'abord le notebook `Projet_ML_complet.ipynb` "
             "(Run All) : il crée le dossier `models/` à côté de ce fichier.")
    st.stop()

pipe, meta = load_artifacts()
THR = meta["threshold"]
FEATURES = meta["features"]
NUM, CAT = meta["numeric"], meta["categorical"]

st.title("Quels clients appeler pour un dépôt à terme ?")
st.caption(f"Modèle : {meta['model_name']} · entraîné sur {meta['n_clients']:,} clients · "
           f"seuil de décision : {THR:.0%}".replace(",", " "))
if meta.get("sklearn_version") != sklearn.__version__:
    st.warning(f"Le modèle a été entraîné avec scikit-learn {meta['sklearn_version']} "
               f"et tu utilises la {sklearn.__version__} : les résultats peuvent différer.")

tab_pred, tab_batch, tab_perf, tab_about = st.tabs(
    ["Un client", "Une liste de clients", "Performance du modèle", "Méthode et limites"])


def decision(p):
    return "À appeler en priorité" if p >= THR else "Priorité faible"


# ───────────────────────── Un client ─────────────────────────
with tab_pred:
    st.write("Renseigne le profil du client : le modèle estime sa probabilité de souscrire.")
    with st.form("client"):
        cols = st.columns(3)
        values = {}
        for i, feat in enumerate(FEATURES):
            with cols[i % 3]:
                if feat in CAT:
                    opts = CAT[feat]["options"]
                    values[feat] = st.selectbox(feat, opts, index=opts.index(CAT[feat]["default"]))
                else:
                    s = NUM[feat]
                    if s["is_int"]:
                        values[feat] = st.number_input(feat, int(s["min"]), int(s["max"]),
                                                       int(round(s["median"])), step=1)
                    else:
                        values[feat] = st.number_input(feat, float(s["min"]), float(s["max"]),
                                                       float(s["median"]), step=0.01, format="%.3f")
        go = st.form_submit_button("Estimer la probabilité", type="primary")

    if go:
        X = pd.DataFrame([values])[FEATURES]
        p = float(pipe.predict_proba(X)[0, 1])
        c1, c2, c3 = st.columns(3)
        c1.metric("Probabilité de souscription", f"{p:.1%}")
        c2.metric("Taux moyen des clients", f"{meta['base_rate']:.1%}")
        c3.metric("Par rapport à la moyenne", f"× {p / meta['base_rate']:.1f}")
        st.progress(min(p, 1.0))
        if p >= THR:
            st.success(f"**{decision(p)}** — la probabilité dépasse le seuil de {THR:.0%}.")
        else:
            st.info(f"**{decision(p)}** — la probabilité reste sous le seuil de {THR:.0%}.")
        st.caption("Une probabilité n'est pas une certitude : elle sert à classer les clients, "
                   "pas à prédire un individu avec sûreté.")

# ───────────────────────── Liste de clients ─────────────────────────
with tab_batch:
    st.write("Charge un fichier CSV : chaque client reçoit une probabilité et il est classé "
             "du plus au moins prometteur.")
    st.caption("Colonnes attendues : " + ", ".join(FEATURES) +
               (" (la colonne `duration`, si présente, est ignorée)." if meta.get("drop_duration") else "."))
    up = st.file_uploader("Fichier CSV", type="csv")
    if up is not None:
        try:
            data = pd.read_csv(up, sep=None, engine="python")
            data.columns = data.columns.str.strip()
        except Exception as e:
            st.error(f"Lecture impossible : {e}")
            st.stop()
        missing = [c for c in FEATURES if c not in data.columns]
        if missing:
            st.error("Colonnes manquantes : " + ", ".join(missing))
        else:
            probas = pipe.predict_proba(data[FEATURES])[:, 1]
            out = data.copy()
            out.insert(0, "probabilite", probas.round(4))
            out.insert(1, "decision", [decision(p) for p in probas])
            out = out.sort_values("probabilite", ascending=False).reset_index(drop=True)
            n_prio = int((probas >= THR).sum())
            a, b = st.columns(2)
            a.metric("Clients dans le fichier", f"{len(out):,}".replace(",", " "))
            b.metric("À appeler en priorité", f"{n_prio:,} ({n_prio / len(out):.0%})".replace(",", " "))
            st.dataframe(out, width="stretch", height=380)
            st.download_button("Télécharger les résultats (CSV)",
                               out.to_csv(index=False).encode("utf-8"),
                               "clients_scores.csv", "text/csv")

# ───────────────────────── Performance ─────────────────────────
with tab_perf:
    m = meta["metrics_test"]
    st.subheader("Résultats sur des clients jamais vus pendant l'entraînement")
    c = st.columns(5)
    c[0].metric("Precision", f"{m['precision']:.0%}", help="Parmi les clients signalés, part qui souscrit vraiment.")
    c[1].metric("Recall", f"{m['recall']:.0%}", help="Parmi les vrais souscripteurs, part détectée.")
    c[2].metric("F1", f"{m['f1']:.2f}")
    c[3].metric("ROC-AUC", f"{m['roc_auc']:.2f}")
    c[4].metric("PR-AUC", f"{m['pr_auc']:.2f}", help=f"Un modèle au hasard obtiendrait environ {meta['base_rate']:.2f}.")

    left, right = st.columns(2)
    with left:
        st.subheader("Gain de ciblage")
        g = meta["gains"]
        gdf = pd.DataFrame({"Clients appelés": list(g), "Souscripteurs captés": [f"{v:.0%}" for v in g.values()]})
        st.dataframe(gdf, hide_index=True, width="stretch")
        st.caption("Au hasard, appeler 20 % des clients capterait 20 % des souscripteurs.")
    with right:
        st.subheader("Matrice de confusion")
        cm = meta["confusion_matrix"]
        st.dataframe(pd.DataFrame([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
                                  index=["Réel : non", "Réel : oui"],
                                  columns=["Prédit : non", "Prédit : oui"]), width="stretch")

    st.subheader("Variables les plus influentes")
    st.bar_chart(pd.Series(meta["importances"]).sort_values())

    st.subheader("Les 7 modèles comparés")
    comp = pd.DataFrame(meta["comparison"]).set_index("modele")
    st.dataframe(comp[["cv_f1", "precision", "recall", "f1", "roc_auc", "pr_auc"]].round(3),
                 width="stretch")

# ───────────────────────── À propos ─────────────────────────
with tab_about:
    st.markdown(f"""
**Données.** Campagne de télémarketing d'une banque (UCI Bank Marketing, Moro et al., 2014).
Seulement {meta['base_rate']:.0%} des clients souscrivent : les classes sont très déséquilibrées.

**Méthode.** Séparation train/test stratifiée, puis pré-traitement, SMOTE et modèle réunis dans un
pipeline pour éviter toute fuite d'information. Sept modèles comparés avec `GridSearchCV`,
choix du modèle sur la validation croisée, seuil de décision réglé sur des prédictions out-of-fold.

**Variable exclue.** {"`duration` (durée de l'appel) est retirée : elle n'est connue qu'après l'appel, donc inutilisable pour décider qui appeler." if meta.get("drop_duration") else "`duration` est conservée : attention, elle n'est pas connue avant l'appel."}

**Limites.**
- Données d'une seule banque sur une période donnée.
- Pas de validation chronologique.
- Le seuil optimise le F1, pas un coût réel en euros.
- Outil de démonstration pédagogique, pas un outil de décision en production.
""")
