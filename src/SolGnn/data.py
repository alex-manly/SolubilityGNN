from typing import Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from rdkit.Chem import MolFromInchi, AddHs
from rdkit.Chem.rdchem import Mol

from SolGnn.config import SplitConfig


class Dataset:
    """
    Splits my dataset into train/validation/test sets in a radom but repeatable way
    as defined by the supplied SplitConfig settings
    """
    def __init__(self, full_dataset: pd.DataFrame, split_config: SplitConfig) -> None:

        self.split_config = split_config
        self.X_train, X_tmp, self.y_train, y_tmp = train_test_split( # get training data
            full_dataset.values[:,0],
            full_dataset.values[:,1],
            test_size = 1- self.split_config.train_ratio,
            random_state = self.split_config.random_state
        )
        self.X_valid, self.X_test, self.y_valid, self.y_test = train_test_split( # split remaining into validation and test sets
            X_tmp,
            y_tmp,
            test_size = self.split_config.test_ratio / (self.split_config.test_ratio + self.split_config.val_ratio),
            random_state = self.split_config.random_state
        )

        # scaler transformer fit only on training data - no data leakage
        self.y_scaler = StandardScaler().fit(self.y_train.reshape(-1,1)) # z-scaling

        self.y_valid_scaled = self.y_scaler.transform(self.y_valid.reshape(-1, 1))
        self.y_train_scaled = self.y_scaler.transform(self.y_train.reshape(-1,1))
        self.y_test_scaled = self.y_scaler.transform(self.y_test.reshape(-1, 1))

def get_dataset(csv_path : Path, split_config : Optional[SplitConfig] = None) -> Dataset:
    full_dataset = pd.read_csv(csv_path, index_col=0)
    if split_config is None:
        split_config = SplitConfig()
    return Dataset(full_dataset, split_config)

def inchi_to_mol(inchi: str) -> Mol:
    """ This is key to repeatably creating our molecular graphs
    rdkit likes to inplicitly represent hydrogens in molecules
    I want to include the hydrogens
    Using AddHs also recovers a few molecules that would otherwise
    have no bonds and be removed from our dataset
    """
    return AddHs(MolFromInchi(inchi, removeHs=False))
