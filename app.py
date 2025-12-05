import streamlit as st
import pandas as pd
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import create_model, Field
from typing import Optional

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="ShieldFlow Enterprise", page_icon="🛡️", layout="wide")

# --- 2. GESTION DE L'ÉTAT ---
if "custom_fields" not in st.session_state:
    st.session_state["custom_fields"] = [
        {"name": "full_name", "desc": "Prénom et Nom corrigés"},
        {"name": "email", "desc": "Email valide"},
        {"name": "company", "desc": "Nom de l'entreprise"},
    ]

if "batch_results" not in st.session_state:
    st.session_state["batch_results"] = None

# --- 3. FONCTIONS CORE ---

def create_dynamic_model(fields_list):
    """Crée le modèle Pydantic avec le Score de Confiance intégré."""
    field_definitions = {
        "risk_flag": (bool, Field(description="Vrai si risqué (spam, fake)")),
        "risk_reason": (Optional[str], Field(description="Raison du risque")),
        "confidence_score": (int, Field(description="Score de 0 à 100 sur la qualité."))
    }
    for field in fields_list:
        field_definitions[field["name"]] = (Optional[str], Field(description=field["desc"]))
    return create_model('DynamicContact', **field_definitions)

def get_traffic_light(score, threshold):
    if score >= threshold: return "🟢 VALIDÉ", "success"
    elif score < 40: return "🔴 REJETÉ", "error"
    else: return "🟠 À RÉVISER", "warning"

def apply_json_mapping(data_dict, template_str):
    """
    Remplace les placeholders {{cle}} dans le template par les valeurs.
    """
    if not template_str.strip():
        return data_dict # Pas de template, on renvoie le plat
    
    try:
        # On travaille sur la chaîne de caractères pour le remplacement simple
        mapped_str = template_str
        for key, value in data_dict.items():
            # Gestion des valeurs None/Null pour le JSON
            val_str = str(value) if value is not None else ""
            # Remplacement du placeholder {{key}}
            mapped_str = mapped_str.replace(f"{{{{{key}}}}}", val_str)
        
        # On tente de parser pour vérifier que c'est du JSON valide
        return json.loads(mapped_str)
    except json.JSONDecodeError:
        return {"error": "Template JSON invalide", "raw_data": data_dict}
    except Exception as e:
        return {"error": str(e), "raw_data": data_dict}

# --- 4. INTERFACE ---

st.title("🛡️ ShieldFlow Enterprise")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. API KEY
    api_key = None
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = st.text_input("Clé API OpenAI", type="password")

    st.divider()
    
    # 2. TRAFFIC LIGHT
    st.subheader("🚦 Seuils")
    confidence_threshold = st.slider("Seuil d'acceptation", 50, 100, 80)
    
    st.divider()
    
    # 3. SCHEMA BUILDER (Champs à extraire)
    st.subheader("🏗️ Champs à Extraire")
    with st.expander("Gérer les champs", expanded=False):
        with st.form("add_field"):
            new_name = st.text_input("Nom (ex: phone)")
            new_desc = st.text_input("Desc (ex: Format E.164)")
            if st.form_submit_button("Ajouter"):
                st.session_state["custom_fields"].append({"name": new_name, "desc": new_desc})
                st.rerun()
        
        st.markdown("---")
        for i, field in enumerate(st.session_state["custom_fields"]):
            c1, c2 = st.columns([5,1])
            c1.text(field['name'])
            if c2.button("🗑️", key=f"d{i}"):
                st.session_state["custom_fields"].pop(i)
                st.rerun()

    st.divider()

    # 4. MAPPING DE SORTIE (NOUVEAU !)
    st.subheader("📤 Mapping de Sortie")
    use_mapping = st.checkbox("Activer le format personnalisé")
    
    json_template = ""
    if use_mapping:
        default_template = """{
  "meta": {
    "source": "ShieldFlow",
    "score": "{{confidence_score}}"
  },
  "crm_data": {
    "contact_name": "{{full_name}}",
    "contact_email": "{{email}}",
    "company": "{{company}}"
  }
}"""
        json_template = st.text_area(
            "Template JSON (Utilisez {{nom_champ}})", 
            value=default_template,
            height=250,
            help="Définissez la structure exacte que votre CRM attend."
        )

if not api_key:
    st.stop()

# INIT IA
DynamicModel = create_dynamic_model(st.session_state["custom_fields"])
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
structured_llm = llm.with_structured_output(DynamicModel)

# --- TABS ---
tab1, tab2 = st.tabs(["⚡ Test Unitaire", "📂 Batch Processing"])

# === TAB 1 : UNITAIRE ===
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        raw_text = st.text_area("Input", height=150, placeholder="Martin, martin@airbus.com")
        run_btn = st.button("Analyser", type="primary")
    
    with col2:
        if run_btn and raw_text:
            with st.spinner("Extraction & Mapping..."):
                # 1. Extraction
                field_names = ", ".join([f["name"] for f in st.session_state["custom_fields"]])
                system_prompt = f"Extrais : {field_names}. Calcule confidence_score."
                prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", raw_text)])
                chain = prompt | structured_llm
                res = chain.invoke({})
                res_dict = res.dict()

                # 2. Mapping (Si activé)
                final_output = res_dict
                if use_mapping:
                    final_output = apply_json_mapping(res_dict, json_template)

                # 3. Affichage
                status_text, status_color = get_traffic_light(res_dict["confidence_score"], confidence_threshold)
                st.metric("Score Qualité", f"{res_dict['confidence_score']}/100", status_text)
                
                st.markdown("### Résultat Final (JSON)")
                st.json(final_output)

# === TAB 2 : BATCH ===
with tab2:
    st.markdown("### Import CSV")
    uploaded_file = st.file_uploader("Fichier CSV", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(3))
        
        if st.button("Lancer Batch 🚀"):
            results = []
            mapped_results = [] # Pour le JSON final
            
            progress_bar = st.progress(0)
            rows = df.head(5).to_dict(orient="records") # Limite démo
            
            for i, row in enumerate(rows):
                # Extraction
                row_text = str(row)
                field_names = ", ".join([f["name"] for f in st.session_state["custom_fields"]])
                system_prompt = f"Extrais : {field_names}. Calcule confidence_score."
                
                prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input_data}")])
                chain = prompt | structured_llm
                res = chain.invoke({"input_data": row_text})
                res_dict = res.dict()
                
                # Ajout Statut pour le tableau de review
                status, _ = get_traffic_light(res_dict["confidence_score"], confidence_threshold)
                res_dict["STATUS"] = status
                results.append(res_dict)
                
                # Mapping pour l'export JSON
                if use_mapping:
                    mapped_results.append(apply_json_mapping(res_dict, json_template))
                
                progress_bar.progress((i + 1) / len(rows))
            
            st.session_state["batch_results"] = pd.DataFrame(results)
            st.session_state["mapped_json"] = mapped_results # On stocke le JSON mappé
            st.success("Terminé !")

    # REVIEW & EXPORT
    if st.session_state["batch_results"] is not None:
        st.divider()
        st.subheader("Validation")
        
        # 1. Tableau d'édition (On garde les données plates pour faciliter la correction humaine)
        cols = ['STATUS', 'confidence_score'] + [f['name'] for f in st.session_state["custom_fields"]]
        # On s'assure que les colonnes existent dans le dataframe
        valid_cols = [c for c in cols if c in st.session_state["batch_results"].columns]
        edited_df = st.data_editor(st.session_state["batch_results"][valid_cols], num_rows="dynamic", use_container_width=True)
        
        col_dl1, col_dl2 = st.columns(2)
        
        # --- LOGIQUE D'EXPORT INTELLIGENTE ---
        
        # Si le mapping est activé, on doit régénérer les données mappées basées sur les corrections de l'utilisateur
        if use_mapping:
            # On prend les données corrigées par l'utilisateur
            corrected_records = edited_df.to_dict(orient="records")
            final_mapped_list = []
            
            for record in corrected_records:
                # On ré-applique le template sur la donnée corrigée
                mapped_record = apply_json_mapping(record, json_template)
                final_mapped_list.append(mapped_record)
            
            # A. Export JSON (Facile)
            json_str = json.dumps(final_mapped_list, indent=2)
            col_dl2.download_button("📥 JSON (Mappé)", json_str, "shieldflow_custom.json", "application/json")
            
            # B. Export CSV (Complexe : Il faut aplatir le JSON imbriqué)
            # Fonction locale pour aplatir
            def flatten_json(y):
                out = {}
                def flatten(x, name=''):
                    if type(x) is dict:
                        for a in x:
                            flatten(x[a], name + a + '_')
                    elif type(x) is list:
                        i = 0
                        for a in x:
                            flatten(a, name + str(i) + '_')
                            i += 1
                    else:
                        out[name[:-1]] = x
                flatten(y)
                return out

            # On aplatit chaque enregistrement mappé
            flat_list = [flatten_json(item) for item in final_mapped_list]
            df_export = pd.DataFrame(flat_list)
            
            csv_mapped = df_export.to_csv(index=False).encode('utf-8')
            col_dl1.download_button("📥 CSV (Mappé & Aplatit)", csv_mapped, "shieldflow_mapped.csv", "text/csv")
            
        else:
            # Pas de mapping, export simple des données corrigées
            csv = edited_df.to_csv(index=False).encode('utf-8')
            col_dl1.download_button("📥 CSV (Standard)", csv, "shieldflow.csv", "text/csv")