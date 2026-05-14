from rdkit import Chem
from rdkit.Chem import Descriptors

smiles = "O=C1c2ccccc2C(=O)c3c1ccc(O)c3O" # Alizarina
mol = Chem.MolFromSmiles(smiles)

features = {
    "Peso_Molecular": Descriptors.MolWt(mol),
    "LogP": Descriptors.MolLogP(mol),
    "Polaridad_TPSA": Descriptors.TPSA(mol),
    "Donantes_H": Descriptors.NumHDonors(mol),
    "Aceptores_H": Descriptors.NumHAcceptors(mol),
    "Anillos_Aromaticos": Descriptors.NumAromaticRings(mol),
    "Acidos_Carboxilicos": Descriptors.fr_COO(mol),
    "Hidroxilos_Fenolicos": Descriptors.fr_Ar_OH(mol), # Típico en pigmentos
    "Aminas_Primarias": Descriptors.fr_NH2(mol)
}

print(features)