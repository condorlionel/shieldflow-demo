import streamlit as st
import time
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Optional

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="ShieldFlow Hybrid Core", page_icon="🛡️", layout="wide")

# --- 2. MOTEUR HYBRIDE (FONCTIONS DE PERFORMANCE) ---

# NIVEAU 1 : Validation Regex (Ultra-Rapide < 1ms)
def quick_validate_email(text: str) -> bool:
    """Vérifie s'il y a au moins un semblant d'email dans le texte."""
    # Regex simple : quelque chose @ quelque chose . quelque chose
    match = re.search(r"[^@]+@[^@]+\.[^@]+", text)
    return bool(match)

# NIVEAU 2 : Cache (Simulation Redis avec Session State)
if "cache_db" not in st.session_state:
    st.session_state["cache_db"] = {}

def check_cache(raw_text: str):
    """Vérifie si on a déjà traité cette demande exacte."""
    return st.session_state["cache_db"].get(raw_text)

def save_to_cache(raw_text: str, result_data: dict):
    """Sauvegarde le résultat pour la prochaine fois."""
    st.session_state["cache_db"][raw_text] = result_data

# --- 3. MODÈLE DE DONNÉES ---
class CleanedContact(BaseModel):
    full_name: Optional[str] = Field(description="Prénom et Nom corrigés")
    email: Optional[str] = Field(description="Email valide")
    job_title: Optional[str] = Field(description="Titre du poste original")
    standardized_role: Optional[str] = Field(description="Rôle standardisé (ex: CEO, Sales)")
    company_name: Optional[str] = Field(description="Nom de l'entreprise")
    company_industry: Optional[str] = Field(description="Secteur d'activité")
    risk_flag: bool = Field(description="Vrai si risqué")
    risk_reason: Optional[str] = Field(description="Raison du risque")
    processing_source: str = Field(description="Source du traitement: 'CACHE' ou 'AI'")

# --- 4. INTERFACE ---
st.title("🛡️ ShieldFlow Core")
st.caption("Architecture Hybride : Regex -> Cache -> IA")

# Gestion Clé API
api_key = None
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = st.sidebar.text_input("Clé API OpenAI", type="password")

if not api_key:
    st.warning("Entrez une clé API pour activer le Niveau 3 (IA).")
    st.stop()

# Initialisation IA (Lazy loading)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
structured_llm = llm.with_structured_output(CleanedContact)

# --- 5. ZONE DE TEST ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📥 Input")
    raw_text = st.text_area("Donnée brute", height=200, placeholder="Ex: martin@airbus..com")
    run_btn = st.button("Lancer le traitement ⚡", type="primary")

with col2:
    st.markdown("### 📤 Output & Performance")
    
    if run_btn and raw_text:
        start_time = time.time()
        final_result = None
        step_log = []

        # --- ÉTAPE 1 : REGEX (The Gatekeeper) ---
        step_log.append("1️⃣ Regex Check...")
        if not quick_validate_email(raw_text):
            # REJET IMMÉDIAT
            end_time = time.time()
            duration = (end_time - start_time) * 1000
            st.error(f"❌ Rejeté par le Niveau 1 (Pas d'email détecté). Temps: {duration:.2f}ms")
            st.stop()
        
        # --- ÉTAPE 2 : CACHE (The Memory) ---
        step_log.append("2️⃣ Cache Check...")
        cached_result = check_cache(raw_text)
        
        if cached_result:
            # HIT CACHE
            final_result = cached_result
            final_result['processing_source'] = "CACHE (Redis)"
            step_log.append("✅ Trouvé en cache !")
        else:
            # --- ÉTAPE 3 : IA (The Brain) ---
            step_log.append("3️⃣ AI Processing (GPT-4o-mini)...")
            try:
                system_prompt = "Tu es ShieldFlow. Nettoie cette donnée B2B. Sois précis."
                prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", raw_text)])
                chain = prompt | structured_llm
                
                res = chain.invoke({})
                final_result = res.dict()
                final_result['processing_source'] = "AI (Generative)"
                
                # Mise en cache pour la prochaine fois
                save_to_cache(raw_text, final_result)
                
            except Exception as e:
                st.error(f"Erreur IA: {e}")
                st.stop()

        # --- RÉSULTATS ---
        end_time = time.time()
        total_duration = (end_time - start_time) * 1000 # en ms
        
        # Affichage du Chrono
        if total_duration < 500:
            st.success(f"⏱️ Temps Total : **{total_duration:.0f} ms** (Ultra-Rapide)")
        elif total_duration < 1500:
            st.warning(f"⏱️ Temps Total : **{total_duration:.0f} ms** (Standard IA)")
        else:
            st.error(f"⏱️ Temps Total : **{total_duration:.0f} ms** (Lent)")

        # Affichage des étapes
        st.caption(" > ".join(step_log))
        
        # Affichage JSON
        st.json(final_result)