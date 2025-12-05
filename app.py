import streamlit as st
import pandas as pd
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import create_model, Field
from typing import Optional, List

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="ShieldFlow Enterprise", page_icon="🛡️", layout="wide")

# --- 2. GESTION DE L'ÉTAT ---
if "custom_fields" not in st.session_state:
    st.session_state["custom_fields"] = [
        {"name": "full_name", "desc": "Prénom et Nom corrigés"},
        {"name": "email", "desc": "Email valide"},
        {"name": "company", "desc": "Nom de l'entreprise"},
        {"name": "job_title", "desc": "Poste standardisé"},
    ]

if "batch_results" not in st.session_state:
    st.session_state["batch_results"] = None

# --- 3. FONCTIONS CORE ---

def create_dynamic_model(fields_list):
    """Crée le modèle Pydantic avec le Score de Confiance intégré."""
    field_definitions = {
        "risk_flag": (bool, Field(description="Vrai si risqué (spam, fake)")),
        "risk_reason": (Optional[str], Field(description="Raison du risque")),
        # LE CŒUR DU TRAFFIC LIGHT SYSTEM :
        "confidence_score": (int, Field(description="Score de 0 à 100 sur la qualité et la complétude de la donnée."))
    }
    
    for field in fields_list:
        field_definitions[field["name"]] = (Optional[str], Field(description=field["desc"]))
    
    return create_model('DynamicContact', **field_definitions)

def get_traffic_light(score, threshold):
    """Détermine la couleur en fonction du seuil."""
    if score >= threshold:
        return "🟢 VALIDÉ", "success"
    elif score < 40: # Seuil critique fixe pour le rouge
        return "🔴 REJETÉ", "error"
    else:
        return "🟠 À RÉVISER", "warning"

# --- 4. INTERFACE ---

st.title("🛡️ ShieldFlow Enterprise")

# --- SIDEBAR : CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. CLÉ API
    api_key = None
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = st.text_input("Clé API OpenAI", type="password")

    st.divider()
    
    # 2. TRAFFIC LIGHT SETTINGS
    st.subheader("🚦 Seuils de Validation")
    confidence_threshold = st.slider(
        "Seuil d'acceptation", 
        min_value=50, max_value=100, value=80,
        help="Score minimum pour être validé (Vert)."
    )
    
    st.divider()
    
    # 3. SCHEMA BUILDER (Gestion Complète)
    st.subheader("🏗️ Structure de Données")
    
    # A. Formulaire d'Ajout
    with st.expander("➕ Ajouter un champ", expanded=False):
        with st.form("add_field_form"):
            new_name = st.text_input("Nom (ex: budget)")
            new_desc = st.text_input("Description (ex: Budget max)")
            submitted = st.form_submit_button("Ajouter")
            
            if submitted and new_name and new_desc:
                st.session_state["custom_fields"].append({"name": new_name, "desc": new_desc})
                st.rerun()

    # B. Liste des Champs Actifs (Visualiser & Supprimer)
    st.markdown("#### Champs Actifs :")
    
    # On itère sur une copie de la liste pour éviter les bugs d'index pendant la suppression
    for i, field in enumerate(st.session_state["custom_fields"]):
        col_a, col_b = st.columns([5, 1])
        
        # Affichage du nom et description en tooltip
        col_a.markdown(f"**{field['name']}**")
        col_a.caption(f"_{field['desc']}_")
        
        # Bouton de suppression
        if col_b.button("🗑️", key=f"del_{i}"):
            st.session_state["custom_fields"].pop(i)
            st.rerun()

    # C. Bouton Reset
    if st.button("🔄 Reset Standard B2B"):
        st.session_state["custom_fields"] = [
            {"name": "full_name", "desc": "Prénom et Nom corrigés"},
            {"name": "email", "desc": "Email valide"},
            {"name": "company", "desc": "Nom de l'entreprise"},
            {"name": "job_title", "desc": "Poste standardisé"},
        ]
        st.rerun()
        
if not api_key:
    st.warning("Veuillez entrer une clé API.")
    st.stop()

# INITIALISATION IA
DynamicModel = create_dynamic_model(st.session_state["custom_fields"])
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
structured_llm = llm.with_structured_output(DynamicModel)

# --- TABS : UNITAIRE vs BATCH ---
tab1, tab2 = st.tabs(["⚡ Test Unitaire (Temps Réel)", "📂 Batch & Review (Fichiers)"])

# === TAB 1 : TEST UNITAIRE ===
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        raw_text = st.text_area("Input", height=150, placeholder="Martin, martin@airbus..com")
        run_btn = st.button("Analyser", type="primary")
    
    with col2:
        if run_btn and raw_text:
            with st.spinner("Analyse..."):
                # Prompt
                field_names = ", ".join([f["name"] for f in st.session_state["custom_fields"]])
                system_prompt = f"""Tu es un auditeur de données. Extrais: {field_names}.
                Evalue la qualité de la donnée (confidence_score 0-100).
                Si l'email est invalide ou le nom manquant, baisse le score."""
                
                prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", raw_text)])
                chain = prompt | structured_llm
                res = chain.invoke({})
                
                # Traffic Light Logic
                status_text, status_color = get_traffic_light(res.confidence_score, confidence_threshold)
                
                # Affichage Visuel
                st.metric("Score de Confiance", f"{res.confidence_score}/100", status_text)
                
                if status_color == "success":
                    st.success("Donnée prête pour export CRM")
                elif status_color == "warning":
                    st.warning("Nécessite une validation humaine")
                else:
                    st.error("Donnée critique / Spam")
                
                st.json(res.dict())

# === TAB 2 : BATCH & REVIEW ===
with tab2:
    st.markdown("### Import CSV & Nettoyage en masse")
    
    uploaded_file = st.file_uploader("Uploader un CSV (Max 5 lignes pour la démo)", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(), use_container_width=True)
        
        if st.button("Lancer le traitement Batch 🚀"):
            results = []
            progress_bar = st.progress(0)
            
            # On limite à 5 lignes pour éviter de vider ton crédit OpenAI en test
            rows_to_process = df.head(5).to_dict(orient="records")
            
            for i, row in enumerate(rows_to_process):
                # Conversion de la ligne en texte
                row_text = str(row)
                
                # Appel IA
                field_names = ", ".join([f["name"] for f in st.session_state["custom_fields"]])
                system_prompt = f"Extrais et nettoie : {field_names}. Calcule confidence_score."
                
                # --- CORRECTION ICI ---
                # 1. On définit un placeholder {input_data} dans le prompt
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt), 
                    ("human", "{input_data}") 
                ])
                
                chain = prompt | structured_llm
                
                # 2. On injecte la vraie donnée via invoke
                # LangChain va remplacer {input_data} par row_text proprement
                res = chain.invoke({"input_data": row_text})
                # ----------------------
                
                # On aplatit le résultat pour le tableau
                res_dict = res.dict()
                
                # Calcul du statut
                status_text, _ = get_traffic_light(res_dict["confidence_score"], confidence_threshold)
                res_dict["STATUS"] = status_text 
                
                results.append(res_dict)
                progress_bar.progress((i + 1) / len(rows_to_process))

            # Stockage en session pour l'édition
            st.session_state["batch_results"] = pd.DataFrame(results)
            st.success("Traitement terminé !")

    # --- INTERFACE DE REVIEW (DATA EDITOR) ---
    if st.session_state["batch_results"] is not None:
        st.divider()
        st.subheader("🕵️ Interface de Validation (Human-in-the-loop)")
        st.info("Corrigez les lignes 'À RÉVISER' directement dans le tableau ci-dessous.")
        
        # On met la colonne STATUS en premier
        cols = ['STATUS', 'confidence_score'] + [f['name'] for f in st.session_state["custom_fields"]] + ['risk_flag', 'risk_reason']
        df_results = st.session_state["batch_results"][cols]
        
        # LE TABLEAU ÉDITABLE
        edited_df = st.data_editor(
            df_results,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "confidence_score": st.column_config.ProgressColumn(
                    "Confiance",
                    help="Score de qualité",
                    format="%d",
                    min_value=0,
                    max_value=100,
                ),
                "STATUS": st.column_config.TextColumn(
                    "Statut",
                    help="Vert = OK, Orange = Review",
                    validate="^(🟢 VALIDÉ|🟠 À RÉVISER|🔴 REJETÉ)$"
                )
            }
        )
        
        # BOUTON D'EXPORT FINAL
        csv = edited_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger les données propres (CSV)",
            data=csv,
            file_name='shieldflow_cleaned.csv',
            mime='text/csv',
        )