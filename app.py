import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Optional, List

# 1. Configuration de la page
st.set_page_config(page_title="ShieldFlow Demo", page_icon="🛡️", layout="wide")

st.title("🛡️ ShieldFlow.io")
st.subheader("Transformez le chaos en données structurées.")
st.markdown("Collez n'importe quel texte (signature d'email, note de réunion, ligne CRM sale) et voyez l'IA le nettoyer et l'enrichir.")

# 2. Définition de la structure de sortie souhaitée (Le Schema)
class CleanedContact(BaseModel):
    full_name: Optional[str] = Field(description="Prénom et Nom corrigés (Title Case)")
    email: Optional[str] = Field(description="Email valide et corrigé si nécessaire")
    job_title: Optional[str] = Field(description="Titre du poste original")
    standardized_role: Optional[str] = Field(description="Rôle standardisé en Anglais (ex: CEO, CTO, Sales Manager)")
    company_name: Optional[str] = Field(description="Nom de l'entreprise")
    company_industry: Optional[str] = Field(description="Secteur d'activité déduit de l'entreprise")
    risk_flag: bool = Field(description="Vrai si la donnée semble fausse ou spam")
    risk_reason: Optional[str] = Field(description="Pourquoi c'est risqué (ex: email jetable, nom incohérent)")

# 3. Configuration de l'IA (Nécessite une clé API OpenAI dans les secrets ou input)
api_key = st.sidebar.text_input("Votre Clé API OpenAI", type="password")

if api_key:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    structured_llm = llm.with_structured_output(CleanedContact)

    # 4. Interface Utilisateur
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📥 Donnée Brute (Input)")
        raw_text = st.text_area("Collez votre texte ici...", height=300, 
                                placeholder="Ex: c'est martin.gros@airbus..com directeur achat")
        analyze_btn = st.button("Nettoyer & Enrichir ✨", type="primary")

    with col2:
        st.markdown("### 📤 Donnée ShieldFlow (API Output)")
        
        if analyze_btn and raw_text:
            with st.spinner("Analyse ShieldFlow en cours..."):
                try:
                    # Le Prompt magique
                    system_prompt = """Tu es ShieldFlow, une API experte en nettoyage de données B2B.
                    Analyse le texte suivant. Extrais les informations, corrige les fautes de frappe dans les emails,
                    déduis le secteur d'activité de l'entreprise, et standardise le poste.
                    Si l'email contient des erreurs évidentes (ex: gmai.com), corrige-les."""
                    
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", raw_text),
                    ])
                    
                    chain = prompt | structured_llm
                    result = chain.invoke({})
                    
                    # Affichage du résultat
                    st.json(result.dict())
                    
                    if result.risk_flag:
                        st.error(f"⚠️ Risque détecté : {result.risk_reason}")
                    else:
                        st.success("✅ Donnée validée et enrichie")
                        
                except Exception as e:
                    st.error(f"Erreur : {e}")

else:
    st.warning("Veuillez entrer une clé API OpenAI dans la barre latérale pour tester la démo.")

# Footer
st.markdown("---")
st.markdown("© 2025 ShieldFlow.io - API de nettoyage de données par IA.")