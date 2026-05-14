import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import base64
from io import BytesIO

from logic import is_chromophore, predict, RANGES

# Importamos tu lógica (puedes copiar las funciones aquí o importarlas si están en otro archivo)
# Por brevedad, asumo que las funciones: is_chromophore, extract_features, score_feature y predict 
# están definidas arriba o en un archivo llamado logic.py

# --- COPIA AQUÍ TUS FUNCIONES ORIGINALES (is_chromophore, predict, etc.) ---
# [Inserte aquí el código que proporcionaste en tu pregunta]
# --------------------------------------------------------------------------

def get_image_base64(mol):
    """Convierte una molécula RDKit a imagen base64 para mostrar en tablas."""
    if mol is None: return ""
    img = Draw.MolToImage(mol, size=(150, 150))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Configuración de página
st.set_page_config(page_title="BioPigment Predictor", layout="wide")

st.title("🧪 Predictor de Compatibilidad de Biopigmentos")
st.markdown("""
Esta herramienta evalúa si una molécula califica como colorante (Etapa 1: Filtro Cromóforo) 
y su afinidad técnica con distintos tipos de fibras (Etapa 2: Scoring).
""")

# Barra lateral para configuración
st.sidebar.header("Configuración")
min_score = st.sidebar.slider("Puntaje mínimo para considerar 'Apto'", 0, 100, 50)

# Entrada de datos
tab1, tab2 = st.tabs(["Individual (SMILES)", "Carga Masiva (Lista)"])

with tab1:
    smiles_input = st.text_input("Ingrese el SMILES de la molécula:", "O=C1c2ccccc2C(=O)c2c1cc(O)c(O)c2")
    mol_name = st.text_input("Nombre de la molécula (opcional):", "Alizarina")
    
    if st.button("Analizar Molécula"):
        mol = Chem.MolFromSmiles(smiles_input)
        if mol:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(Draw.MolToImage(mol, size=(300, 300)), caption=mol_name)
            
            with col2:
                scores, reason = predict(smiles_input)
                if scores:
                    st.success(f"✅ {reason}")
                    # Crear un DataFrame para mostrar los resultados
                    df_scores = pd.DataFrame([
                        {"Fibra": k, "Score (%)": v, "Estado": "✅ Apto" if v >= min_score else "❌ No Apto"} 
                        for k, v in scores.items()
                    ]).sort_values("Score (%)", ascending=False)
                    
                    st.table(df_scores)
                else:
                    st.error(f"🚫 Descartado: {reason}")
        else:
            st.error("SMILES Inválido")

with tab2:
    st.markdown("Pegue varios SMILES (uno por línea):")
    bulk_input = st.text_area("Listado de SMILES", "O=C1/C(=C2\\Nc3ccccc3C2=O)Nc2ccccc12\nNCC(=O)O\nCCCCCC1=C/C(=C\\C2=C(C=C(N2)C3=CC=CN3)OC)/N=C1C")
    
    if st.button("Procesar Lista"):
        lines = [line.strip() for line in bulk_input.split("\n") if line.strip()]
        results = []
        
        for s in lines:
            scores, reason = predict(s)
            res_dict = {"SMILES": s, "Estado_Cromoforo": reason}
            
            if scores:
                # Añadir los scores de cada tela al diccionario
                res_dict.update(scores)
                # Determinar mejor categoría
                best_fabric = max(scores, key=scores.get)
                res_dict["Mejor Opción"] = f"{best_fabric} ({scores[best_fabric]}%)"
            else:
                for fab in RANGES.keys(): res_dict[fab] = 0
                res_dict["Mejor Opción"] = "N/A"
            
            results.append(res_dict)
        
        df_results = pd.DataFrame(results)
        st.dataframe(df_results.style.highlight_max(axis=1, subset=list(RANGES.keys()), color='#90ee90'))
        
        # Opción de descarga
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button("Descargar Reporte CSV", csv, "resultados_biopigmentos.csv", "text/csv")

# Sección de referencia de rangos
with st.expander("Ver Criterios de Selección (Rangos Químicos)"):
    st.json(RANGES)