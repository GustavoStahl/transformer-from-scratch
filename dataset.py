import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Optional
from torch.utils.data import Dataset

class ChatTask2024(Dataset):
    def __init__(self, root_dir: Union[str,Path], split: str, source_language: str, first_n_samples: Optional[int] = None):

        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        df = pd.read_csv(root_dir / split / "en-pt.csv")
        df = df[:first_n_samples]
        source, target = self.extract_source_target(df, source_language)

        self.source = source
        self.target = target

    def extract_source_target(self, df: pd.DataFrame, source_language: str) -> tuple[np.ndarray, np.ndarray]:
        mask = df["source_language"] == source_language

        # source where source_language == source_language
        source1: np.ndarray = df[mask]["source"].to_numpy()
        target1: np.ndarray = df[mask]["reference"].to_numpy()

        nmask = ~mask
        
        # reference where source_language == target_language
        source2 = df[nmask]["reference"].to_numpy()
        target2 = df[nmask]["source"].to_numpy()

        source = np.append(source1, source2, axis=0)
        target = np.append(target1, target2, axis=0)

        return source, target        

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        return self.source[index], self.target[index]