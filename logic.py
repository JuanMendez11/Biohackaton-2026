"""
Predictor de Compatibilidad de Biopigmentos por Tipo de Tela
============================================================
Pipeline completo: Filtro Cromoforo (Etapa 1) + Score por Tela (Etapa 2)

Uso:
    from predict import predict
    result = predict("CCO")
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# ETAPA 1: Filtro de Cromoforo (SMARTS)
# Si la molecula no absorbe luz visible, no tiene sentido
# evaluarla como colorante.
# ============================================================
CHROMOPHORE_PATTERNS = [
    ("[#6]=[#6]-[#6]=[#6]-[#6]=[#6]", "Sistema conjugado extendido"),
    ("c1cccc2ccccc12", "Naftaleno"),
    ("c1ccc2c(c1)C(=O)c3ccccc3C2=O", "Antraquinona"),
    ("N=Nc1ccccc1", "Azo-benceno"),
    ("c1cc(=O)oc2ccccc12", "Cromona/Flavona"),
    ("c1ccc(-c2ccccc2)cc1", "Bifenilo"),
    ("c1ccc2[nH]ccc2c1", "Indol"),
    ("c1ccc2c(c1)ccc1ccccc12", "Fenantreno"),
]

def is_chromophore(smiles):
    """Verifica si la molecula tiene subestructuras cromoforas o suficiente conjugacion."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "SMILES invalido"

    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())

    # Chequear patrones cromoforos conocidos
    for smarts, name in CHROMOPHORE_PATTERNS:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            return True, f"Cromoforo detectado: {name}"

    # Regla general: >= 2 anillos aromaticos implica absorcion
    if aromatic_rings >= 2:
        return True, f"Cromoforo probable: {aromatic_rings} anillos aromaticos"

    # >= 8 atomos aromaticos (puede ser 1 anillo grande fusionado)
    if aromatic_atoms >= 8:
        return True, f"Cromoforo probable: {aromatic_atoms} atomos aromaticos"

    return False, f"No es cromoforo ({aromatic_rings} anillos, {aromatic_atoms} at. aromaticos)"

# ============================================================
# ETAPA 2: Score por Tela (Rangos del equipo de Quimica)
# ============================================================
RANGES = {
    "Celulosica": {"LogP": (0.5, 3.0), "MolWt": (200, 600), "TPSA": (60, 140), "H_Donors": (2, 6), "H_Acceptors": (3, 8), "Anillos_Arom": (2, 4)},
    "Proteica":   {"LogP": (0.5, 2.5), "MolWt": (200, 600), "TPSA": (80, 160), "H_Donors": (2, 6), "H_Acceptors": (3, 9), "Anillos_Arom": (2, 5)},
    "Artificial":  {"LogP": (0.5, 3.0), "MolWt": (200, 600), "TPSA": (60, 140), "H_Donors": (2, 5), "H_Acceptors": (3, 8), "Anillos_Arom": (2, 4)},
    "Sintetica":  {"LogP": (3.0, 5.5), "MolWt": (400, 800), "TPSA": (20, 55), "H_Donors": (0, 2), "H_Acceptors": (1, 4), "Anillos_Arom": (2, 5)},
}

FEATURES = ["LogP", "MolWt", "TPSA", "H_Donors", "H_Acceptors", "Anillos_Arom"]

FEATURE_LABELS = {
    "LogP": "Hidrofobicidad (LogP)",
    "MolWt": "Peso Molecular",
    "TPSA": "Polaridad (TPSA)",
    "H_Donors": "Donores Puente H",
    "H_Acceptors": "Aceptores Puente H",
    "Anillos_Arom": "Anillos Aromaticos",
}

def extract_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "LogP": Descriptors.MolLogP(mol),
        "MolWt": Descriptors.MolWt(mol),
        "TPSA": Descriptors.TPSA(mol),
        "H_Donors": float(rdMolDescriptors.CalcNumLipinskiHBD(mol)),
        "H_Acceptors": float(rdMolDescriptors.CalcNumLipinskiHBA(mol)),
        "Anillos_Arom": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
    }

def score_feature(val, lo, hi):
    """
    Score individual (0-100). Dentro = 100. Fuera = decaimiento exponencial.
    Mas agresivo que penalizacion lineal: caer fuera del rango castiga fuerte.
    """
    if lo <= val <= hi:
        return 100.0
    range_width = hi - lo if hi > lo else 1.0
    if val < lo:
        distance_out = lo - val
    else:
        distance_out = val - hi
    ratio = distance_out / range_width
    return 100.0 * np.exp(-2.0 * ratio)  # decae rapido

def predict(smiles):
    """
    Pipeline completo. Devuelve None si no es cromoforo.
    Si es cromoforo, devuelve {tela: score} usando media geometrica.
    """
    is_chrom, reason = is_chromophore(smiles)
    if not is_chrom:
        return None, reason

    feats = extract_features(smiles)
    if feats is None:
        return None, "SMILES invalido"

    scores = {}
    for fabric, fabric_ranges in RANGES.items():
        feature_scores = []
        for f in FEATURES:
            lo, hi = fabric_ranges[f]
            feature_scores.append(score_feature(feats[f], lo, hi))
        # Media geometrica: un score malo arrastra todo el promedio
        product = 1.0
        for s in feature_scores:
            product *= max(s, 0.01) / 100.0  # evitar log(0)
        scores[fabric] = round((product ** (1.0 / len(feature_scores))) * 100, 1)

    return scores, "Cromoforo confirmado"

def predict_verbose(smiles, name="Molecula"):
    feats = extract_features(smiles)
    if feats is None:
        print(f"\n[ERROR] SMILES invalido para: {name}")
        return None

    # ETAPA 1: Filtro cromoforo
    is_chrom, reason = is_chromophore(smiles)

    print(f"\n{'='*60}")
    print(f" {name}")
    print(f"{'='*60}")
    print(f"  LogP={feats['LogP']:.2f}, MolWt={feats['MolWt']:.1f}, "
          f"TPSA={feats['TPSA']:.1f}, Donors={int(feats['H_Donors'])}, "
          f"Acceptors={int(feats['H_Acceptors'])}, AromRings={int(feats['Anillos_Arom'])}")
    print(f"  Etapa 1: {reason}")

    if not is_chrom:
        print(f"  >>> DESCARTADO: No absorbe luz visible. No es un colorante.")
        return None

    # ETAPA 2: Score por tela
    scores, _ = predict(smiles)

    for fab, sc in sorted(scores.items(), key=lambda x: -x[1]):
        bar = "#" * int(sc / 2)
        problems = []
        for f in FEATURES:
            lo, hi = RANGES[fab][f]
            val = feats[f]
            if val < lo:
                problems.append(f"    [X] {FEATURE_LABELS[f]}: {val:.2f} (bajo, rango: {lo:.1f}-{hi:.1f})")
            elif val > hi:
                problems.append(f"    [X] {FEATURE_LABELS[f]}: {val:.2f} (alto, rango: {lo:.1f}-{hi:.1f})")

        if sc >= 90:
            status = "EXCELENTE"
        elif sc >= 70:
            status = "BUENO"
        elif sc >= 50:
            status = "PARCIAL"
        else:
            status = "NO APTO"

        print(f"\n  {fab:<12}: {sc:>5.1f}% {bar}  [{status}]")
        for p in problems:
            print(p)

    return scores

if __name__ == "__main__":
    print("\n" + "="*60)
    print(" BIOPIGMENTOS EXITOSOS")
    print("="*60)
    print(predict("OC1=CC(O)=C2C(=O)C(O)=C(OC2=C1)C1=CC(O)=C(O)C=C1"))
    predict_verbose("OC1=CC(O)=C2C(=O)C(O)=C(OC2=C1)C1=CC(O)=C(O)C=C1", "Quercetina (algodon/lana)")
    predict_verbose("O=C1/C(=C2\\Nc3ccccc3C2=O)Nc2ccccc12", "Indigo (algodon)")
    predict_verbose("COC1=CC(/C=C/C(=O)CC(=O)/C=C/C2=CC(OC)=C(O)C=C2)=CC=C1O", "Curcumina (lana/seda)")
    predict_verbose("O=C1c2ccccc2C(=O)c2c1cc(O)c(O)c2", "Alizarina (algodon/lana)")
    predict_verbose("CCCCCC1=C/C(=C\\C2=C(C=C(N2)C3=CC=CN3)OC)/N=C1C", "Prodigiosina (sintetica)")

    print("\n\n" + "="*60)
    print(" MOLECULAS QUE NO SON COLORANTES")
    print("="*60)
    predict_verbose("CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C", "Penicilina (antibiotico)")
    predict_verbose("CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirina (farmaco)")
    predict_verbose("C(C(CO)O)O", "Glicerol (solvente)")
    predict_verbose("NCC(=O)O", "Glicina (aminoacido)")