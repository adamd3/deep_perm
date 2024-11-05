import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def analyze_chemical_similarity(smiles_train, smiles_val, smiles_test):
    """Analyze chemical similarity between splits"""

    def get_fingerprints(smiles_list):
        fps = []
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
                fps.append(fp)
        return fps

    # Get fingerprints for each set
    train_fps = get_fingerprints(smiles_train)
    val_fps = get_fingerprints(smiles_val)
    test_fps = get_fingerprints(smiles_test)

    def calc_avg_sim(fps1, fps2):
        sims = []
        for fp1 in fps1[:100]:  # Sample 100 for efficiency
            for fp2 in fps2[:100]:
                sim = DataStructs.TanimotoSimilarity(fp1, fp2)
                sims.append(sim)
        return np.mean(sims)

    return {
        "train_val_similarity": calc_avg_sim(train_fps, val_fps),
        "train_test_similarity": calc_avg_sim(train_fps, test_fps),
        "val_test_similarity": calc_avg_sim(val_fps, test_fps),
    }
