from dataclasses import dataclass
from typing import Callable, List, Dict, Optional, Sequence, Tuple
from rdkit.Chem.rdchem import Mol
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import itertools
import torch
from torch_geometric.data import Data
import numpy as np

from SolGnn.data import inchi_to_mol

@dataclass(frozen=True)
class AtomFeatureSpec:
    one_hot: List[str]
    scaled: List[str]

@dataclass(frozen=True)
class BondFeatureSpec:
    one_hot: List[str]
    scaled: List[str]

@dataclass(frozen=True)
class GlobalFeatureSpec:
    scaled: List[Callable[[Mol], float]]

class FeatureBasis:

    """
    An sklearn-style data preprocesser. We need to one-hot-encode some features
    (think atomic symbol and bond type) but scale other (e.g. molecular weight).
    This basis will be fitted on a list of rdkit Mol's from our training dataset,
    then be able to spit out graph features for our graph representation with
    concatenated feature vectors.
    """

    def __init__(
            self,
            atoms: AtomFeatureSpec,
            bonds: BondFeatureSpec,
            global_: GlobalFeatureSpec,
    ):
        self.atoms = atoms
        self.bonds = bonds
        self.global_ = global_

        self._atom_ohe: Dict[str, OneHotEncoder] = {}
        self._atom_scaler: Optional[StandardScaler] = None

        self._bond_scaler: Optional[StandardScaler] = None
        self._bond_ohe: Dict[str, OneHotEncoder] = {}

        self._global_scaler: Optional[StandardScaler] = None

        self._fitted = False

    def fit(self, inchis: Sequence[str]) -> "FeatureBasis":
        mols = [inchi_to_mol(inchi) for inchi in inchis]
        # ----- fit one-hot-encoders for specifed atom features
        for feat in (self.atoms.one_hot or []):  # go through each feature that we need to one-hot-encode
            vals = []
            for mol in mols:
                for atom in mol.GetAtoms():
                    v = getattr(atom, feat)()
                    vals.append([v])
            enc = OneHotEncoder(handle_unknown="ignore",
                                sparse_output=False)  # sparse isn't necessary... better for debugging to see what's going on
            enc.fit(np.array(vals))  # fit encoder
            self._atom_ohe[feat] = enc  # save as this basis

        # ----- fit standard scalers for specified atom features
        atom_num_feats = (self.atoms.scaled or [])
        if atom_num_feats:  # we can handle only passing OHE features
            rows = []
            for mol in mols:
                for atom in mol.GetAtoms():
                    rows.append([float(getattr(atom, f)()) for f in atom_num_feats])
            self._atom_scaler = StandardScaler()
            self._atom_scaler.fit(np.array(rows, dtype=float))  # fit the scaler

        # ----- fit one-hot-encoders for specifed bond features
        for feat in (self.bonds.one_hot or []):  # go through each feature that we need to one-hot-encode
            vals = []
            for mol in mols:
                for bond in mol.GetBonds():
                    v = getattr(bond, feat)()
                    vals.append([v])
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            enc.fit(np.array(vals))  # fit encoder
            self._bond_ohe[feat] = enc  # save as this basis

        # ----- fit scaler across all speified bonds -----
        bond_feats = (self.bonds.scaled or [])
        if bond_feats:
            rows = []
            for mol in mols:
                for bond in mol.GetBonds():
                    rows.append([float(getattr(bond, f)()) for f in bond_feats])
            self._bond_scaler = StandardScaler()
            self._bond_scaler.fit(np.array(rows, dtype=float))

        # ----- global: fit scaler across all mols -----
        glob_funcs = (self.global_.scaled or [])
        if glob_funcs:
            rows = []
            for mol in mols:
                rows.append([float(fn(mol)) for fn in glob_funcs])
            self._global_scaler = StandardScaler()
            self._global_scaler.fit(np.array(rows, dtype=float))

        self._fitted = True
        return self

    def transform(self, inchis: Sequence[str], ys: Optional[Sequence[float]] = None) -> List[Data]:
        self._require_fitted()

        # If ys is None, create an iterator that yields None for each inchi
        ys_iter = [torch.as_tensor(y) for y in ys] if ys is not None else itertools.repeat(None)

        if ys is not None and len(ys) != len(inchis):
            raise ValueError(f"Length mismatch: got {len(inchis)} inchis but {len(ys)} targets.")

        data_list = []
        for inchi, y in zip(inchis, ys_iter):
            mol = inchi_to_mol(inchi)
            atom_features = self.featurize_atoms(mol)
            edge_index, bond_features = self.featurize_bonds(mol)
            global_features = self.featurize_global(mol)

            # Data takes torch.tensor as inputs
            atom_features = torch.as_tensor(atom_features)
            edge_index = torch.as_tensor(edge_index)
            edge_attr = torch.as_tensor(bond_features)
            global_feat = torch.as_tensor(global_features)

            data_list.append(Data(
                x=atom_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                global_feat=global_feat,
                y=y
            ))

        return data_list


    # --------- graph feature transform functinns
    def featurize_atoms(self, mol: Mol) -> np.ndarray:
        self._require_fitted()

        out_cols = []

        # one-hot-encoded atom features
        for feat in (self.atoms.one_hot or []):
            enc = self._atom_ohe[feat]
            vals = np.array([[getattr(atom, feat)()] for atom in mol.GetAtoms()], dtype=object)
            out_cols.append(enc.transform(vals))

        # scaled atomic features
        atom_num_feats = (self.atoms.scaled or [])
        if atom_num_feats:
            X = np.array(
                [[float(getattr(atom, f)()) for f in atom_num_feats] for atom in mol.GetAtoms()],
                dtype=float,
            )
            Xs = self._atom_scaler.transform(X)
            out_cols.append(Xs)

        return np.concatenate(out_cols, axis=1)

    def featurize_bonds(self, mol: Mol) -> Tuple[np.ndarray, np.ndarray]:
        self._require_fitted()

        bonds = list(mol.GetBonds())
        n_bonds = len(bonds)

        # --- edge_index: (2, 2*n_bonds) ---
        edge_pairs = []
        for b in bonds:
            i = b.GetBeginAtomIdx()
            j = b.GetEndAtomIdx()
            edge_pairs.append([i, j])
            edge_pairs.append([j, i])
        edge_index = np.array(edge_pairs, dtype=np.int64).T

        # --- build per-(undirected)-bond feature matrix: (n_bonds, d_edge) ---
        blocks = []

        # One-hot encoded bond features
        for feat in (self.bonds.one_hot or []):
            enc = self._bond_ohe[feat]

            raw = [getattr(b, feat)() for b in bonds]
            try:
                vals = np.asarray(raw, dtype=float).reshape(-1, 1)
            except (TypeError, ValueError):
                vals = np.asarray(raw, dtype=object).reshape(-1, 1)

            ohe = enc.transform(vals)  # shape: (n_bonds, d_ohe)
            blocks.append(ohe)

        # Scaled bond features
        bond_num_feats = (self.bonds.scaled or [])
        if bond_num_feats:
            X = np.array(
                [[float(getattr(b, f)()) for f in bond_num_feats] for b in bonds],
                dtype=np.float32,
            )  # shape: (n_bonds, k)
            Xs = self._bond_scaler.transform(X)  # shape: (n_bonds, k)
            blocks.append(Xs)

        if not blocks:
            # No bond features requested
            edge_attr = np.zeros((2 * n_bonds, 0), dtype=np.float32)
            return edge_index, edge_attr

        bond_feat = np.concatenate(blocks, axis=1)  # (n_bonds, d_edge)

        # --- duplicate to align with directed edges ---
        # Since edge_index is [(i,j),(j,i)] per bond in that order, repeat rows the same way.
        edge_attr = np.repeat(bond_feat, repeats=2, axis=0).astype(np.float32)  # (2*n_bonds, d_edge)

        return edge_index, edge_attr

    def featurize_global(self, mol: Mol) -> np.ndarray:
        self._require_fitted()

        glob_funcs = (self.global_.scaled or [])
        if not glob_funcs:
            return np.zeros((0,), dtype=float)

        X = np.array([[float(fn(mol)) for fn in glob_funcs]], dtype=float)  # (1, n)
        Xs = self._global_scaler.transform(X)
        return Xs

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("FeatureBasis is not fitted. Call .fit(training_mols) first.")