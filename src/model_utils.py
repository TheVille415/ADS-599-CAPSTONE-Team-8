import json
from pathlib import Path

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, Crippen, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# Streamlit app configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "outputs" / "artifacts"

# silence noisy RDKit parse warnings in terminal
RDLogger.DisableLog("rdApp.error")

class SmilesLSTMTuned(torch.nn.Module):
    def __init__(self, vocab_size, pad_idx, emb_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = torch.nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        logits = self.fc(out)
        return logits

# -----------------------------
# Model loading and inference
# -----------------------------
def load_model_assets(
    model_path=ARTIFACT_DIR / "smiles_lstm_tuned.pt",
    meta_path=ARTIFACT_DIR / "smiles_lstm_tuned_meta.json",
):
    with open(meta_path, "r") as f:
        meta = json.load(f)

    stoi = meta["stoi"]
    itos = {int(k): v for k, v in meta["itos"].items()}
    max_len = meta["max_len"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SmilesLSTMTuned(
        vocab_size=len(stoi),
        pad_idx=stoi[meta["pad_token"]],
        emb_dim=meta["emb_dim"],
        hidden_dim=meta["hidden_dim"],
        num_layers=meta["num_layers"],
        dropout=meta["dropout"],
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return {
        "model": model,
        "stoi": stoi,
        "itos": itos,
        "max_len": max_len,
        "device": device,
        "start_token": meta["start_token"],
        "end_token": meta["end_token"],
        "pad_token": meta["pad_token"],
    }

# Tuned LSTM generation functiongic
def sample_smiles_tuned(assets, temperature=0.7):
    model = assets["model"]
    stoi = assets["stoi"]
    itos = assets["itos"]
    max_len = assets["max_len"]
    device = assets["device"]
    start_token = assets["start_token"]
    end_token = assets["end_token"]
    pad_token = assets["pad_token"]

    current = torch.tensor([[stoi[start_token]]], dtype=torch.long).to(device)
    generated = []
    hidden = None

    with torch.no_grad():
        for _ in range(max_len):
            emb = model.embedding(current[:, -1:])
            out, hidden = model.lstm(emb, hidden)
            logits = model.fc(out[:, -1, :]) / temperature
            probs = torch.softmax(logits, dim=-1)

            next_idx = torch.multinomial(probs, num_samples=1).item()
            next_char = itos[next_idx]

            if next_char == end_token:
                break
            if next_char != pad_token:
                generated.append(next_char)

            current = torch.cat(
                [current, torch.tensor([[next_idx]], dtype=torch.long).to(device)],
                dim=1
            )

    return "".join(generated)

# Validation Helper Function
def canonical_if_valid(smiles: str):
    if not smiles or not smiles.strip():
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True)

# Valid generation loop with retries
def generate_valid_analog(assets, attempts=100, temperature=0.45, min_length=5):
    """
    Sample repeatedly until a valid canonical SMILES is found.
    Returns None if no valid candidate is produced.
    """
    for _ in range(attempts):
        candidate = sample_smiles_tuned(assets, temperature=temperature)

        if not candidate or len(candidate.strip()) < min_length:
            continue

        valid = canonical_if_valid(candidate)
        if valid is not None:
            return valid

    return None

# -----------------------------
# Molecular properties and comparison
# -----------------------------
def molecule_properties(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return {
        "Formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular Weight": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Crippen.MolLogP(mol), 2),
        "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "H Bond Donors": Lipinski.NumHDonors(mol),
        "H Bond Acceptors": Lipinski.NumHAcceptors(mol),
        "Ring Count": Lipinski.RingCount(mol),
        "Heavy Atom Count": Lipinski.HeavyAtomCount(mol),
    }


_fpgen = GetMorganGenerator(radius=2, fpSize=2048)

# Compute Tanimoto similarity between two SMILES strings
def tanimoto_similarity(smiles_a: str, smiles_b: str):
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return None

    fp_a = _fpgen.GetFingerprint(mol_a)
    fp_b = _fpgen.GetFingerprint(mol_b)
    return round(DataStructs.TanimotoSimilarity(fp_a, fp_b), 3)