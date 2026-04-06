import streamlit as st
#from main import PAGE
from utils.theme import  (is_dark,toggle_theme,BLEU, VERT, ORANGE, GRIS)

# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
def kpi(label, val, delta="", cls=""):
    st.markdown(f"""<div class="kpi {cls}">
      <div class="kl">{label}</div><div class="kv">{val}</div>
      <div class="kd">{delta}</div></div>""", unsafe_allow_html=True)

def sh(t): st.markdown(f'<div class="sh">{t}</div>', unsafe_allow_html=True)
def box(t, cls=""): st.markdown(f'<div class="box {cls}">{t}</div>', unsafe_allow_html=True)
def commentaire(t): st.markdown(f'<div class="commentaire"> <strong>{t}</strong></div>', unsafe_allow_html=True)


JOURS_FR   = {"Monday":"Lundi","Tuesday":"Mardi","Wednesday":"Mercredi",
               "Thursday":"Jeudi","Friday":"Vendredi","Saturday":"Samedi","Sunday":"Dimanche"}
MOIS_FR    = {1:"Jan",2:"Fév",3:"Mar",4:"Avr",5:"Mai",6:"Jun",
              7:"Jul",8:"Aoû",9:"Sep",10:"Oct",11:"Nov",12:"Déc"}
JOURS_ORD  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
JOURS_FR_ORD = [JOURS_FR[j] for j in JOURS_ORD]
MOIS_ORD   = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]

# Couleurs texte/fond adaptées au thème courant
T_PRIMARY   = "#E2E8F0" if is_dark() else "#0F172A"
T_SECONDARY = "#94A3B8" if is_dark() else "#6B7280"
BG_CARD     = "#1A2535" if is_dark() else "#FFFFFF"
GRID_COLOR  = "#2C3E50" if is_dark() else "#F1F5F9"

def plo(**kwargs):
    """Retourne un dict de layout Plotly adapté au thème."""
    base = dict(
        plot_bgcolor=BG_CARD,
        paper_bgcolor=BG_CARD,
        margin=dict(t=20,b=0,l=0,r=0),
        font=dict(color=T_SECONDARY, family="Inter, sans-serif"),
        yaxis=dict(gridcolor=GRID_COLOR, color=T_SECONDARY, zerolinecolor=GRID_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR, color=T_SECONDARY, zerolinecolor=GRID_COLOR),
    )
    base.update(kwargs)
    return base
CHART      = dict(plot_bgcolor="white", paper_bgcolor="white",
                  margin=dict(t=20,b=0,l=0,r=0),
                  yaxis=dict(gridcolor="#F1F5F9"),
                  xaxis=dict(gridcolor="#F1F5F9"))

#=============================================================
# SIDEBAR
#=============================================================
def render_sidebar():
    
    # ──────────────────────────────────────────────────────────────
    # SESSION STATE — mode d'utilisation
    # ──────────────────────────────────────────────────────────────
    if "mode" not in st.session_state:
        st.session_state.mode = None   # None = écran de sélection


    # ──────────────────────────────────────────────────────────────
    # ÉCRAN DE SÉLECTION D'OBJECTIF
    # ──────────────────────────────────────────────────────────────
    if st.session_state.mode is None:
        # Sidebar minimale sur l'écran de sélection
        with st.sidebar:
            st.markdown(f"""
            <div style='text-align:center;padding:24px 0 16px;'>
            <div style='font-size:2.8rem;'>🚗</div>
            <div style='font-size:1.3rem;font-weight:700;color:#F1F5F9;margin-top:6px;'>TrafficML</div>
            <div style='font-size:.65rem;color:#64748B;letter-spacing:.12em;margin-top:4px;'>
                INTERSTATE 94 · MINNEAPOLIS</div>
            </div>
            <hr style='border-color:#2C3E50;margin:12px 0;'>
            <div style='font-size:.78rem;color:#64748B;padding:8px 4px;line-height:1.7;'>
            Bienvenue sur TrafficML.<br><br>
            Choisissez votre profil pour accéder à l'interface adaptée à vos besoins.
            </div>""", unsafe_allow_html=True)

            theme_icon  = "☀️" if is_dark() else "🌙"
            theme_label = "Mode clair" if is_dark() else "Mode sombre"
            st.markdown("<hr style='border-color:#2C3E50;margin:16px 0 8px;'>", unsafe_allow_html=True)
            if st.button(f"{theme_icon}  {theme_label}", use_container_width=True, key="theme_btn_home"): toggle_theme()

        # ── Page de sélection ──
        st.markdown(f"""
        <div style='text-align:center;padding:40px 0 8px;'>
        <div style='font-size:3.5rem;'>🚗</div>
        <h1 style='font-size:2.6rem;font-weight:700;color:var(--text-primary);margin:12px 0 6px;'>TrafficML</h1>
        <p style='font-size:1.1rem;color:var(--text-secondary);margin:0;'>
            Prédiction du Trafic Urbain · Interstate 94 · Minneapolis-Saint Paul</p>
        <div style='display:flex;justify-content:center;gap:12px;margin-top:16px;flex-wrap:wrap;'>
            <span style='background:{BLEU}22;color:{BLEU};border-radius:20px;padding:4px 14px;font-size:.78rem;font-weight:600;'>Random Forest · R²=0.989</span>
            <span style='background:{VERT}22;color:{VERT};border-radius:20px;padding:4px 14px;font-size:.78rem;font-weight:600;'>RMSE = 210 véh/h</span>
            <span style='background:{ORANGE}22;color:{ORANGE};border-radius:20px;padding:4px 14px;font-size:.78rem;font-weight:600;'>MAPE = 5.8%</span>
        </div>
        </div>
        <hr style='border-color:var(--border);margin:32px 0 28px;'>
        <h2 style='text-align:center;font-size:1.3rem;font-weight:600;color:var(--text-primary);margin-bottom:6px;'>
        Quel est votre objectif ?</h2>
        <p style='text-align:center;color:var(--text-secondary);font-size:.92rem;margin-bottom:32px;'>
        Choisissez le profil qui correspond à votre usage pour accéder à l'interface adaptée.</p>
        """, unsafe_allow_html=True)

        col_gap, c1, c2, col_gap2 = st.columns([0.5, 3, 3, 0.5])

        with c1:
            st.markdown(f"""
            <div style='background:var(--bg-card);border:2px solid {BLEU};border-radius:16px;
                        padding:32px 28px;text-align:center;height:100%;'>
            <div style='font-size:3rem;margin-bottom:12px;'>🎓</div>
            <div style='font-size:1.2rem;font-weight:700;color:{BLEU};margin-bottom:10px;'>Mode Pédagogique</div>
            <div style='font-size:.85rem;color:var(--text-secondary);line-height:1.7;margin-bottom:20px;'>
                Pour les <b>étudiants, enseignants et chercheurs</b> qui souhaitent comprendre
                la démarche complète de data science : exploration, preprocessing, modélisation,
                évaluation et interprétabilité.
            </div>
            <div style='text-align:left;font-size:.8rem;color:var(--text-secondary);margin-bottom:24px;'>
                {''.join([f"<div style='padding:4px 0;'>✅ {p}</div>" for p in [
                "Exploration & visualisation (EDA)",
                "Feature engineering documenté",
                "Comparaison des 3 modèles",
                "Interprétabilité SHAP détaillée",
                "Prédiction interactive",
                "Conclusions & perspectives"
                ]])}
            </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🎓  Accéder au mode pédagogique", use_container_width=True,
                        key="btn_peda", type="primary"):
                st.session_state.mode = "pedagogique"
                st.rerun()

        with c2:
            st.markdown(f"""
            <div style='background:var(--bg-card);border:2px solid {VERT};border-radius:16px;
                        padding:32px 28px;text-align:center;height:100%;'>
            <div style='font-size:3rem;margin-bottom:12px;'>💼</div>
            <div style='font-size:1.2rem;font-weight:700;color:{VERT};margin-bottom:10px;'>Mode Professionnel</div>
            <div style='font-size:.85rem;color:var(--text-secondary);line-height:1.7;margin-bottom:20px;'>
                Pour les <b>décideurs, opérationnels et professionnels</b> qui ont besoin
                de résultats directs : performances des modèles et prédictions en temps réel,
                sans les détails techniques.
            </div>
            <div style='text-align:left;font-size:.8rem;color:var(--text-secondary);margin-bottom:24px;'>
                {''.join([f"<div style='padding:4px 0;'>✅ {p}</div>" for p in [
                "Tableau de bord des performances",
                "Comparaison synthétique des modèles",
                "Prédiction interactive avancée",
                "Visualisations claires et épurées",
                "Interface rapide et efficace",
                "Focus sur les résultats métier"
                ]])}
            </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("💼  Accéder au mode professionnel", use_container_width=True,
                        key="btn_pro"):
                st.session_state.mode = "pro"
                st.rerun()

        # Footer
        st.markdown(f"""
        <div style='text-align:center;margin-top:40px;color:var(--text-muted);font-size:.75rem;'>
        Saidou Yameogo · Interstate 94 · Minneapolis-Saint Paul · 2024<br>
        <span style='color:{BLEU};'>Random Forest</span> ·
        <span style='color:{VERT};'>XGBoost</span> ·
        <span style='color:{GRIS};'>Ridge</span>
        </div>""", unsafe_allow_html=True)
        st.stop()


    # ──────────────────────────────────────────────────────────────
    # SIDEBAR — selon le mode sélectionné
    # ──────────────────────────────────────────────────────────────
    MODE = st.session_state.mode   # "pedagogique" ou "pro"

    with st.sidebar:
        # Logo + titre
        mode_badge_color = BLEU if MODE == "pedagogique" else VERT
        mode_badge_label = "🎓 Pédagogique" if MODE == "pedagogique" else "💼 Professionnel"
        st.markdown(f"""
        <div style='text-align:center;padding:18px 0 10px;'>
        <div style='font-size:2.2rem;'>🚗</div>
        <div style='font-size:1.2rem;font-weight:700;color:#F1F5F9;'>TrafficML</div>
        <div style='font-size:.65rem;color:#94A3B8;letter-spacing:.1em;margin-top:3px;'>
            INTERSTATE 94</div>
        <div style='background:{mode_badge_color}33;color:{mode_badge_color};border-radius:20px;
                    padding:3px 12px;font-size:.7rem;font-weight:600;margin-top:8px;display:inline-block;'>
            {mode_badge_label}</div>
        </div>
        <hr style='border-color:#2C3E50;margin:8px 0 10px;'>""", unsafe_allow_html=True)

        # Toggle thème
        theme_icon  = "☀️" if is_dark() else "🌙"
        theme_label = "Mode clair" if is_dark() else "Mode sombre"
        if st.button(f"{theme_icon}  {theme_label}", use_container_width=True, key="theme_btn"): toggle_theme()

        st.markdown("<hr style='border-color:#2C3E50;margin:10px 0 12px;'>", unsafe_allow_html=True)

        # Navigation selon le mode
        if MODE == "pedagogique":
            PAGES_PEDA = [
                "🏠  Accueil",
                "📊  Exploration (EDA)",
                "⚙️  Feature Engineering",
                "🤖  Modélisation",
                "📈  Évaluation & Performances",
                "🔬  Interprétabilité SHAP",
                "🔮  Prédiction Interactive",
                "📝  Conclusions & Perspectives",
            ]
            PAGE = st.radio("Select page", PAGES_PEDA, label_visibility="collapsed")
        else:
            PAGES_PRO = [
                "🏠  Tableau de bord",
                "📈  Performances & Résultats",
                "🔮  Prédiction Interactive",
            ]
            PAGE = st.radio("Select page", PAGES_PRO, label_visibility="collapsed")

        st.markdown("<hr style='border-color:#2C3E50;margin:12px 0;'>", unsafe_allow_html=True)

        # Bouton changer de mode
        other_mode = "professionnel" if MODE == "pedagogique" else "pédagogique"
        other_icon = "💼" if MODE == "pedagogique" else "🎓"
        if st.button(f"{other_icon}  Mode {other_mode}", use_container_width=True, key="switch_mode"):
            st.session_state.mode = None
            st.rerun()

        st.markdown(f"""
        <div style='font-size:.68rem;color:#475569;line-height:1.8;margin-top:8px;'>
        <b style='color:#94A3B8;'>Meilleur modèle</b><br>
        Random Forest · R²=0.989<br>
        RMSE=210 · MAPE=5.8%<br><br>
        <b style='color:#94A3B8;'>Dataset</b><br>
        48 204 obs. · 2012–2018
        </div>""", unsafe_allow_html=True)

    if MODE == "pro":
  
        # ── P-PRO-1 : Tableau de bord ──
        if PAGE == "🏠  Tableau de bord": st.switch_page("pages/dashboard.py")
        elif PAGE == "📈  Performances & Résultats": st.switch_page("pages/resultat_pro.py")
        # ── P-PRO-3 : Prédiction Interactive (Pro) ──
        elif PAGE == "🔮  Prédiction Interactive": st.switch_page("pages/prediction.py")
    else:
        # ── P-PEDA-1 : Accueil ──
        if PAGE == "🏠  Accueil": st.switch_page("pages/accueil.py")
        # ── P-PEDA-2 : Exploration (EDA) ──
        elif PAGE == "📊  Exploration (EDA)": st.switch_page("pages/eda.py")
        # ── P-PEDA-3 : Feature Engineering ──
        elif PAGE == "⚙️  Feature Engineering": st.switch_page("pages/feature_engineering.py")
        # ── P-PEDA-4 : Modélisation ──
        elif PAGE == "🤖  Modélisation": st.switch_page("pages/modelisation.py")
        # ── P-PEDA-5 : Évaluation & Performances ──
        elif PAGE == "📈  Évaluation & Performances": st.switch_page("pages/evaluation.py")
        # ── P-PEDA-6 : Interprétabilité SHAP ──
        elif PAGE == "🔬  Interprétabilité SHAP": st.switch_page("pages/shap.py")
        # ── P-PEDA-7 : Prédiction Interactive (Pédagogique) ──
        elif PAGE == "🔮  Prédiction Interactive": st.switch_page("pages/prediction_peda.py")
        # ── P-PEDA-8 : Conclusions & Perspectives ──
        elif PAGE == "📝  Conclusions & Perspectives": st.switch_page("pages/conclusion.py")