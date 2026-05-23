import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioPredict",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS PERSONNALISÉ
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', "Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", sans-serif;
}

/* ── Fond ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

/* ── Header hero ── */
.hero-block {
    background: linear-gradient(135deg, rgba(220,38,38,0.15) 0%, rgba(239,68,68,0.05) 100%);
    border: 1px solid rgba(220,38,38,0.3);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-block::before {
    content: '❤️';
    position: absolute;
    right: 2rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 6rem;
    opacity: 0.12;
}
.hero-title {
    font-family: 'DM Serif Display', serif !important;
    font-size: 3rem !important; /* Modifiez cette valeur (ex: 3.5rem) */
    color: #ffffff !important;
    margin: 0 0 0.4rem 0 !important;
    line-height: 1.1 !important;
}
.hero-title span { color: #f87171 !important; }
.hero-sub {
    color: #94a3b8 !important;
    font-size: 1rem !important;
    font-weight: 300 !important;
    margin: 0 !important;
}

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.6rem;
    margin-bottom: 1.2rem;
}
.card-title {
    color: #f1f5f9;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Résultat cards ── */
.result-card {
    border-radius: 16px;
    padding: 1.8rem 1.6rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s;
}
.result-card:hover { transform: translateY(-3px); }
.result-card.danger {
    background: linear-gradient(135deg, rgba(220,38,38,0.25) 0%, rgba(185,28,28,0.15) 100%);
    border: 1.5px solid rgba(220,38,38,0.5);
}
.result-card.safe {
    background: linear-gradient(135deg, rgba(16,185,129,0.2) 0%, rgba(5,150,105,0.1) 100%);
    border: 1.5px solid rgba(16,185,129,0.4);
}
.result-card.neutral {
    background: rgba(255,255,255,0.04);
    border: 1.5px solid rgba(255,255,255,0.12);
}
.model-name {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 0.8rem;
}
.result-icon { font-size: 2.8rem; margin-bottom: 0.5rem; }
.result-label {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    font-weight: 400;
    margin-bottom: 0.3rem;
}
.result-label.danger { color: #f87171; }
.result-label.safe   { color: #34d399; }
.result-proba {
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0.5rem 0;
    color: #ffffff;
}
.result-proba.danger { color: #fca5a5; }
.result-proba.safe   { color: #6ee7b7; }
.confidence-badge {
    display: inline-block;
    background: rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    font-size: 0.78rem;
    color: #cbd5e1;
    margin-top: 0.6rem;
}
.confidence-badge span { color: #facc15; font-weight: 600; }

/* ── Barre de probabilité ── */
.proba-bar-wrap {
    background: rgba(255,255,255,0.06);
    border-radius: 8px;
    height: 8px;
    margin-top: 0.9rem;
    overflow: hidden;
}
.proba-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.8s ease;
}

/* ── Verdict global ── */
.verdict-block {
    border-radius: 20px;
    padding: 2rem 2.5rem;
    text-align: center;
    margin: 1.5rem 0;
}
.verdict-block.danger {
    background: linear-gradient(135deg, rgba(220,38,38,0.2), rgba(127,29,29,0.15));
    border: 2px solid rgba(220,38,38,0.4);
}
.verdict-block.safe {
    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(5,150,105,0.1));
    border: 2px solid rgba(16,185,129,0.35);
}
.verdict-icon { font-size: 3.5rem; }
.verdict-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    margin: 0.5rem 0 0.3rem;
    color: #ffffff;
}
.verdict-sub { color: #94a3b8; font-size: 0.95rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
.sidebar-logo {
    text-align: center;
    padding: 1.5rem 0 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1.5rem;
}
.sidebar-logo .logo-icon { font-size: 3rem; }
.sidebar-logo .logo-text {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #f87171 !important;
}
.sidebar-logo .logo-sub {
    font-size: 0.72rem;
    color: #64748b !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Inputs style ── */
.stSlider > div > div > div { background: #1e293b; }
.stNumberInput input, .stSelectbox select {
    background: rgba(30,41,59,0.8) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #f1f5f9 !important;
    border-radius: 8px !important;
}

/* ── Bouton ── */
.stButton > button {
    background: linear-gradient(135deg, #dc2626, #b91c1c) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 2.5rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    width: 100% !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 24px rgba(220,38,38,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(220,38,38,0.45) !important;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ── Info note ── */
.info-note {
    background: rgba(251,191,36,0.08);
    border-left: 3px solid #fbbf24;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: #fcd34d;
    margin-top: 1rem;
}

/* ── Metric pill ── */
.metric-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 30px;
    padding: 0.3rem 0.9rem;
    font-size: 0.8rem;
    color: #cbd5e1;
    margin: 0.2rem;
}
.metric-pill b { color: #f1f5f9; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES MODÈLES ET ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR = "."   # ← Même dossier que app.py — adaptez si nécessaire

# Accuracies obtenues sur le jeu de test (résultats de votre notebook)
ACCURACIES = {
    "XGBoost":               0.7220,
}
AUC_ROCS = {
    "XGBoost":               0.7856,
}

@st.cache_resource
def charger_modeles():
    """Charge les modèles et artefacts une seule fois."""
    try:
        models = {
            "XGBoost":               joblib.load(os.path.join(MODEL_DIR, "model_xgboost.pkl")),
        }
        scaler        = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        lambda_values = joblib.load(os.path.join(MODEL_DIR, "lambda_values.pkl"))
        return models, scaler, lambda_values, None
    except FileNotFoundError as e:
        return None, None, None, str(e)

models, scaler, lambda_values, load_error = charger_modeles()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-icon">❤️</div>
        <div class="logo-text">CardioPredict</div>
        <div class="logo-sub">ML · Cardiovascular Risk</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**À propos**")
    st.markdown("""
    <div style="color:#94a3b8;font-size:0.85rem;line-height:1.7;">
    Application de prédiction du risque cardiovasculaire basée sur un modèle de Machine Learning (XGBoost)
    entraîné sur <b style="color:#e2e8f0">62 502 patients</b>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Modèle utilisé**")

    for nom, acc in ACCURACIES.items():
        icon = "🌲" if "Forest" in nom else ("⚡" if "XGB" in nom else "📌")
        auc  = AUC_ROCS[nom]
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
                    border-radius:10px;padding:0.7rem 1rem;margin-bottom:0.6rem;">
            <div style="font-size:0.8rem;font-weight:600;color:#e2e8f0;">{icon} {nom}</div>
            <div style="display:flex;gap:0.8rem;margin-top:0.3rem;">
                <span style="font-size:0.75rem;color:#94a3b8;">Acc <b style="color:#facc15">{acc:.1%}</b></span>
                <span style="font-size:0.75rem;color:#94a3b8;">AUC <b style="color:#60a5fa">{auc:.4f}</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────
def appliquer_boxcox_manuel(valeur, lam):
    """
    Reproduction exacte de scipy.stats.boxcox avec lambda connu.
    """
    if lam is None:
        return float(valeur)          # age_years → pas de transformation
    
    valeur = float(valeur)
    
    if abs(lam) < 1e-10:             # λ ≈ 0 → transformation log
        return np.log(valeur)
    
    return (np.power(valeur, lam) - 1.0) / lam   # formule exacte SciPy


def preparer_observation(age, poids, taille, ap_hi, ap_lo,
                          cholesterol, gluc, smoke, alco, active,
                          scaler, lambda_values):
    """
    Transforme les entrées brutes de l'utilisateur en vecteur
    prêt pour la prédiction (même pipeline que l'entraînement).
    """
    # 1. Variables synthétiques
    bmi     = poids / (taille / 100) ** 2
    map_val = ap_lo + (ap_hi - ap_lo) / 3.0

    # 2. Box-Cox sur les variables continues (lambda issus de l'entraînement)
    age_t = appliquer_boxcox_manuel(float(age),     lambda_values.get('age_years'))
    bmi_t = appliquer_boxcox_manuel(float(bmi),     lambda_values.get('bmi'))
    map_t = appliquer_boxcox_manuel(float(map_val), lambda_values.get('map'))

    obs = pd.DataFrame({
        'age_years'  : [age_t],
        'bmi'        : [bmi_t],
        'map'        : [map_t],
        'cholesterol': [int(cholesterol)],
        'gluc'       : [int(gluc)],
        'smoke'      : [int(smoke)],
        'alco'       : [int(alco)],
        'active'     : [int(active)],
    })

    # 3. StandardScaler (uniquement sur les variables continues)
    obs[['age_years', 'bmi', 'map']] = scaler.transform(obs[['age_years', 'bmi', 'map']])

    return obs


def couleur_proba(p):
    """Renvoie une couleur CSS interpolée rouge/vert selon la probabilité."""
    r = int(220 * p + 16  * (1 - p))
    g = int(38  * p + 185 * (1 - p))
    b = int(38  * p + 129 * (1 - p))
    return f"rgb({r},{g},{b})"


# ─────────────────────────────────────────────────────────────────────────────
# CORPS PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-block">
    <p class="hero-title">Prédiction du Risque <span>Cardiovasculaire</span></p>
    <p class="hero-sub">Renseignez les données cliniques du patient — l'Intelligence Artificielle analyse son profil.</p>
</div>
""", unsafe_allow_html=True)

if load_error:
    st.error(f"❌ Impossible de charger les modèles : `{load_error}`\n\n"
             f"Placez les fichiers `.pkl` dans le même dossier que `app.py`.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════
# FORMULAIRE D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════
if 'form_key' not in st.session_state:
    st.session_state.form_key = 0

with st.form(f"formulaire_patient_{st.session_state.form_key}"):

    st.markdown("""<div class="card-title">👤 &nbsp;Informations du Patient</div>""",
                unsafe_allow_html=True)

    # ── Ligne 1 : Données anthropométriques ──────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "👤 Âge (années)",
            min_value=30, max_value=70,
            value=50, step=1,
            help="Entre 30 et 70 ans (plage du dataset)"
        )

    with col2:
        poids = st.number_input(
            "📌 Poids (kg)",
            min_value=40.0, max_value=120.0,
            value=75.0, step=0.5,
        )

    with col3:
        taille = st.number_input(
            "📌 Taille (cm)",
            min_value=140.0, max_value=200.0,
            value=170.0, step=0.5,
        )

    # IMC calculé en temps réel
    bmi_affiche = poids / (taille / 100) ** 2
    cat_imc = ("🟢 Normal" if bmi_affiche < 25 else
               "🟡 Surpoids" if bmi_affiche < 30 else "🔴 Obésité")
    st.markdown(
        f'<div style="color:#94a3b8;font-size:0.82rem;margin-bottom:0.8rem;">'
        f'IMC calculé : <b style="color:#f1f5f9">{bmi_affiche:.1f} kg/m²</b> — {cat_imc}'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("""<div class="card-title">🩺 &nbsp;Mesures Cliniques</div>""",
                unsafe_allow_html=True)

    # ── Ligne 2 : Pression artérielle ────────────────────────────
    col4, col5 = st.columns(2)
    with col4:
        ap_hi = st.number_input(
            "🔺 Pression Systolique — ap_hi (mmHg)",
            min_value=90, max_value=180,
            value=120, step=1,
            help="Pression artérielle haute (normale : 90–130 mmHg)"
        )
    with col5:
        ap_lo = st.number_input(
            "🔻 Pression Diastolique — ap_lo (mmHg)",
            min_value=60, max_value=110,
            value=80, step=1,
            help="Pression artérielle basse (normale : 60–90 mmHg)"
        )

    map_affiche = ap_lo + (ap_hi - ap_lo) / 3.0
    st.markdown(
        f'<div style="color:#94a3b8;font-size:0.82rem;margin-bottom:0.8rem;">'
        f'Pression Artérielle Moyenne (PAM) : <b style="color:#f1f5f9">{map_affiche:.1f} mmHg</b>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("""<div class="card-title">🧪 &nbsp;Biomarqueurs</div>""",
                unsafe_allow_html=True)

    col6, col7 = st.columns(2)
    with col6:
        cholesterol = st.selectbox(
            "❤️ Cholestérol",
            options=[1, 2, 3],
            format_func=lambda x: {1: "1 — Normal", 2: "2 — Élevé", 3: "3 — Très Élevé"}[x],
            index=0,
        )
    with col7:
        gluc = st.selectbox(
            "🍬 Glucose",
            options=[1, 2, 3],
            format_func=lambda x: {1: "1 — Normal", 2: "2 — Élevé", 3: "3 — Très Élevé"}[x],
            index=0,
        )

    st.markdown("---")
    st.markdown("""<div class="card-title">🏃 &nbsp;Mode de Vie</div>""",
                unsafe_allow_html=True)

    col8, col9, col10 = st.columns(3)
    with col8:
        smoke = st.radio(
            "🚬 Tabagisme",
            options=[0, 1],
            format_func=lambda x: "Non-fumeur" if x == 0 else "Fumeur",
            horizontal=True,
        )
    with col9:
        alco = st.radio(
            "🍷 Alcool",
            options=[0, 1],
            format_func=lambda x: "Non" if x == 0 else "Oui",
            horizontal=True,
        )
    with col10:
        active = st.radio(
            "🏃 Activité Physique",
            options=[1, 0],
            format_func=lambda x: "Actif" if x == 1 else "Sédentaire",
            horizontal=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("❤️  Lancer la Prédiction")

reset = st.button("❤️  Réinitialiser le Formulaire")

if reset:
    st.session_state.form_key += 1
    st.rerun()

# ═══════════════════════════════════════════════════════════════════
# PRÉDICTIONS ET RÉSULTATS
# ═══════════════════════════════════════════════════════════════════
if submitted:
    with st.spinner("Analyse en cours…"):

        obs = preparer_observation(
            age, poids, taille, ap_hi, ap_lo,
            cholesterol, gluc, smoke, alco, active,
            scaler, lambda_values
        )

       #st.write("Valeurs envoyées au modèle :", obs)

        # S'assurer que les variables catégorielles sont bien des entiers pour XGBoost
        for col in ['cholesterol', 'gluc', 'smoke', 'alco', 'active']:
            obs[col] = obs[col].astype(int)

        model_xgb = models["XGBoost"]
        pred_xgb  = int(model_xgb.predict(obs)[0])
        proba_xgb = float(model_xgb.predict_proba(obs)[0][1])

    st.markdown("---")
    st.markdown("""<div class="card-title" style="font-size:1rem;margin-bottom:1.5rem;">
        📋 &nbsp;Résultats de la Prédiction
    </div>""", unsafe_allow_html=True)

    # ── Verdict global ──────────────────────────────────
    verdict = "danger" if pred_xgb == 1 else "safe"

    if verdict == "danger":
        verdict_icon  = "⚠️"
        verdict_titre = "Risque Cardiovasculaire Élevé"
        verdict_sub   = f"L'IA détecte un risque · Probabilité d'atteinte : {proba_xgb:.1%}"
    else:
        verdict_icon  = "✅"
        verdict_titre = "Profil Cardiovasculaire Favorable"
        verdict_sub   = f"L'IA ne détecte pas de risque · Probabilité d'atteinte : {proba_xgb:.1%}"

    st.markdown(f"""
    <div class="verdict-block {verdict}">
        <div class="verdict-icon">{verdict_icon}</div>
        <div class="verdict-title">{verdict_titre}</div>
        <div class="verdict-sub">{verdict_sub}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Résultats individuels du modèle ───────────────────────
    icons = {
        "XGBoost": "⚡",
    }

    _, col, _ = st.columns([1, 2, 1])
    nom = "XGBoost"
    proba  = proba_xgb
    pred   = pred_xgb
    classe = "danger" if pred == 1 else "safe"
    label  = "Malade" if pred == 1 else "Sain"
    r_icon = "💔" if pred == 1 else "💚"
    acc    = ACCURACIES[nom]
    auc    = AUC_ROCS[nom]
    bar_color = couleur_proba(proba)

    with col:
        st.markdown(f"""
        <div class="result-card {classe}">
            <div class="model-name">{icons[nom]} &nbsp;{nom}</div>
            <div class="result-icon">{r_icon}</div>
            <div class="result-label {classe}">{label}</div>
            <div class="result-proba {classe}">{proba:.1%}</div>
            <div style="font-size:0.75rem;color:#94a3b8;">probabilité de maladie</div>
            <div class="proba-bar-wrap">
                <div class="proba-bar-fill" style="width:{proba*100:.1f}%;background:{bar_color};"></div>
            </div>
            <div class="confidence-badge">
                Confiance : <span>{acc:.1%}</span> acc · <span>{auc:.4f}</span> AUC
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Récapitulatif du profil patient ───────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="card-title">🗂️ &nbsp;Profil analysé</div>""",
                unsafe_allow_html=True)

    bmi_val = poids / (taille / 100) ** 2
    map_val = ap_lo + (ap_hi - ap_lo) / 3.0
    chol_lb = {1: "Normal", 2: "Élevé", 3: "Très Élevé"}
    gluc_lb = {1: "Normal", 2: "Élevé", 3: "Très Élevé"}

    st.markdown(f"""
    <div style="display:flex;flex-wrap:wrap;gap:0.4rem;margin-top:0.5rem;">
        <span class="metric-pill">👤 <b>{age} ans</b></span>
        <span class="metric-pill">📌 <b>{poids} kg</b></span>
        <span class="metric-pill">📌 <b>{taille} cm</b></span>
        <span class="metric-pill">📐 IMC <b>{bmi_val:.1f}</b></span>
        <span class="metric-pill">🔺 ap_hi <b>{ap_hi}</b></span>
        <span class="metric-pill">🔻 ap_lo <b>{ap_lo}</b></span>
        <span class="metric-pill">💉 PAM <b>{map_val:.1f}</b></span>
        <span class="metric-pill">❤️ Chol. <b>{chol_lb[cholesterol]}</b></span>
        <span class="metric-pill">🍬 Gluc. <b>{gluc_lb[gluc]}</b></span>
        <span class="metric-pill">🚬 <b>{"Fumeur" if smoke else "Non-fumeur"}</b></span>
        <span class="metric-pill">🍷 <b>{"Alcool" if alco else "Sans alcool"}</b></span>
        <span class="metric-pill">🏃 <b>{"Actif" if active else "Sédentaire"}</b></span>
    </div>
    """, unsafe_allow_html=True)

    # ── Note médicale ──────────────────────────────────────────────
    st.markdown("""
    <div class="info-note">
        ⚠️ <b>Avertissement</b> : Cette application est un outil académique de démonstration.
        Les prédictions ne remplacent pas un diagnostic médical professionnel.
        Consultez un médecin pour toute décision de santé.
    </div>
    """, unsafe_allow_html=True)