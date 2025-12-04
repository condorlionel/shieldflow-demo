import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Optional

# --- 1. CONFIGURATION DE LA PAGE (DOIT ÊTRE EN PREMIER) ---
st.set_page_config(
    page_title="ShieldFlow Demo",
    page_icon="🛡️",
    layout="wide"
)

# --- 2. GESTION DE LA CLÉ API (SECRETS OU SIDEBAR) ---
api_key = None

# On vérifie d'abord si la clé est dans les secrets de Streamlit Cloud
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    # Sinon, on affiche un champ dans la barre latérale pour la rentrer manuellement
    api_key = st.sidebar.text_input("Votre Clé API OpenAI", type="password")
    if not api_key:
        st.sidebar.warning("Veuillez entrer une clé API pour continuer.")

# --- 3. INTERFACE PRINCIPALE ---
st.title("🛡️ ShieldFlow.io")
st.subheader("Transformez le chaos en données structurées.")
st.markdown(
    """
    Collez n'importe quel texte (signature d'email, note de réunion, ligne CRM sale) 
    et voyez l'IA le nettoyer, le standardiser et l'enrichir en temps réel.
    """
)

# --- 4. DÉFINITION DU MODÈLE DE DONNÉES (SCHEMA) ---
class CleanedContact(BaseModel):
    full_name: Optional[str] = Field(description="Prénom et Nom corrigés et formatés (Title Case)")
    email: Optional[str] = Field(description="Email valide et corrigé si nécessaire (ex: gmai.com -> gmail.com)")
    job_title: Optional[str] = Field(description="Titre du poste original tel qu'il apparait dans le texte")
    standardized_role: Optional[str] = Field(description="Rôle standardisé en Anglais (ex: CEO, CTO, VP Sales, Engineer)")
    company_name: Optional[str] = Field(description="Nom de l'entreprise identifiée")
    company_industry: Optional[str] = Field(description="Secteur d'activité déduit de l'entreprise (ex: SaaS, Retail, Aerospace)")
    risk_flag: bool = Field(description="Mettre à True si la donnée semble fausse, spam, ou insultante")
    risk_reason: Optional[str] = Field(description="Raison du risque si risk_flag est True")

# --- 5. LOGIQUE DE L'APPLICATION ---

if api_key:
    # Initialisation du modèle seulement si la clé est présente
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
        structured_llm = llm.with_structured_output(CleanedContact)
    except Exception as e:
        st.error(f"Erreur de configuration API : {e}")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📥 Donnée Brute (Input)")
        raw_text = st.text_area(
            "Collez votre texte ici...", 
            height=300, 
            placeholder="Exemple : c'est martin.gros@airbus..com directeur achat basé a toulouse"
        )
        analyze_btn = st.button("Nettoyer & Enrichir ✨", type="primary")

    with col2:
        st.markdown("### 📤 Donnée ShieldFlow (API Output)")
        
        if analyze_btn and raw_text:
            with st.spinner("Analyse ShieldFlow en cours..."):
                try:
                    # Le Prompt Système qui guide l'IA
                    system_prompt = """Tu es ShieldFlow, une API experte en nettoyage de données B2B.
                    Analyse le texte suivant avec une précision extrême.
                    1. Extrais les informations de contact.
                    2. Corrige les fautes de frappe évidentes dans les emails (ex: gmai.com, outlok.fr).
                    3. Déduis le secteur d'activité de l'entreprise si possible.
                    4. Standardise le poste en anglais (ex: 'Directeur des ventes' -> 'Sales Director').
                    Si le texte est du spam ou n'a aucun sens, active le risk_flag."""
                    
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", raw_text),
                    ])
                    
                    # Exécution de la chaîne
                    chain = prompt | structured_llm
                    result = chain.invoke({})
                    
                    # Affichage du résultat JSON
                    st.json(result.dict())
                    
                    # Feedback visuel
                    if result.risk_flag:
                        st.error(f"⚠️ Risque détecté : {result.risk_reason}")
                    else:
                        st.success("✅ Donnée validée et enrichie")
                        
                except Exception as e:
                    st.error(f"Une erreur est survenue lors de l'analyse : {e}")

else:
    # Message d'accueil si pas de clé
    st.info("👋 Bienvenue sur la démo ShieldFlow. L'application est prête à démarrer.")
    if "OPENAI_API_KEY" not in st.secrets:
        st.warning("Aucune clé API détectée dans les secrets. Veuillez en entrer une dans la barre latérale.")

# Footer
st.markdown("---")
st.markdown("© 2025 ShieldFlow.io - Intelligent Data Firewall.")