import torch
import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, pkl_path, mode="processed", seq_length=1024):

        self.mode = mode
        self.seq_length = seq_length
        self.scaler = StandardScaler()

        with open(pkl_path, "rb") as f:
            data_dict = pickle.load(f)

        ids = [ids for ids in data_dict.keys()]
        dfs = [data for data in data_dict.values()]
        self.ids, self.dfs = self._preprocess_and_resample(ids, dfs)

    def _preprocess_and_resample(self, id_list, df_list):
        processed_id_list = []
        processed_df_list = []
        for id, df in zip(id_list, df_list):
            df = df.copy()
            if df.shape[0] < 1000:
                continue

            if self.mode == "processed":
                df = df.iloc[:, 1:-11]
                df = df[df[df.columns[5]] >= 0]
                df = df.drop(columns=[df.columns[3]])
                col_name = df.columns[4]
                col = df.pop(col_name)
                df.insert(0, col_name, col)
            elif self.mode == "raw":
                df = df.iloc[:, 1:-6]
                df = df[df[df.columns[0]] >= 0]
            else:
                raise ValueError("mode must be 'raw' or 'processed'")

            df = df.drop_duplicates(keep=False)
            columns = df.columns.tolist()
            df = self.scaler.fit_transform(df) if self.scaler is not None else df
            df = pd.DataFrame(df, columns=columns)

            df_resampled = self._resample_dataframe(df)
            processed_df_list.append(df_resampled)
            processed_id_list.append(id)

        return processed_id_list, processed_df_list

    def _resample_dataframe(self, df):
        new_index = np.linspace(0, 1, self.seq_length)
        old_index = np.linspace(0, 1, len(df))
        df_resampled = pd.DataFrame(
            {col: np.interp(new_index, old_index, df[col].values) for col in df.columns}
        )
        return df_resampled

    def __len__(self):
        return len(self.dfs)
    
    def __getitem__(self, idx):
        df = self.dfs[idx]
        id = self.ids[idx]
        return torch.tensor(df.values, dtype=torch.float32)