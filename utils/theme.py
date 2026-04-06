# utils/theme.py
import streamlit as st

# Couleurs statiques
BLEU   = "#1E6FD9"
VERT   = "#17B897"
ORANGE = "#F4A223"
ROUGE  = "#E8432A"
GRIS   = "#6B7280"

def init_theme():
    """Initialise le thème dans session_state"""
    if "theme" not in st.session_state:
        st.session_state.theme = "light"

def toggle_theme():
    """Bascule entre thème clair et sombre"""
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
    st.rerun()

def is_dark():
    """Retourne True si thème sombre"""
    return st.session_state.get("theme", "light") == "dark"

def load_css():
    """Charge le fichier CSS et applique le thème"""
    with open("assets/style.css", "r") as f:
        css = f.read()
    
    # Injection des variables CSS selon le thème
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    
    # Application de la classe dark sur body si nécessaire
    if is_dark():
        st.markdown("""
        <script>
            document.body.classList.add('dark');
        </script>
        """, unsafe_allow_html=True)