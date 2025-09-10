import torch
import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


class TimeSeriesDataset(object):
    def __init__(self, pkl_path, mode="processed", seq_len=3, prediction_window=1):

        self.mode = mode
        self.seq_len = seq_len
        self.prediction_window = prediction_window
        self.scaler = StandardScaler()

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.infos = [d["info"] for d in data]

        dfs = [d["data"] for d in data]
        self.dfs = self._preprocess_and_resample(dfs)

    def _preprocess_and_resample(self, df_list):
        processed_list = []
        for df in df_list:
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

            #df_resampled = self._resample_dataframe(df)
            processed_list.append(df)

        return processed_list

    # def _resample_dataframe(self, df):
    #     new_index = np.linspace(0, 1, self.seq_len)
    #     old_index = np.linspace(0, 1, len(df))
    #     df_resampled = pd.DataFrame(
    #         {col: np.interp(new_index, old_index, df[col].values) for col in df.columns}
    #     )
    #     return df_resampled
    
    def frame_series(self, dfs):
        """
        Function used to prepare the data for time series prediction
        :return: TensorDataset
        """

        features, target, y_hist = [], [], []

        for X in dfs:
            nb_obs = X.shape[0]
            X = X.values
            for i in range(1, nb_obs - self.seq_len - self.prediction_window):
                features.append(torch.FloatTensor(X[i:i + self.seq_len, :]).unsqueeze(0))
                y_hist.append(torch.FloatTensor(X[i - 1: i + self.seq_len - 1, :]).unsqueeze(0))
                target.append(torch.FloatTensor(X[i + self.seq_len:i + self.seq_len + self.prediction_window, :]))

        features_var = torch.cat(features)
        y_hist_var = torch.cat(y_hist)
        target_var = torch.cat(target)
        
        return TensorDataset(features_var, y_hist_var, target_var)
    
    def get_loaders(self, batch_size, train_split):
        """
        :return: DataLoaders associated to training and testing data
        """
        nb_features = self.dfs[0].shape[1]

        n_total = len(self.dfs)
        n_train = int(train_split * n_total)
        train_dfs = self.dfs[:n_train]
        val_dfs = self.dfs[n_train:]

        train_dataset = self.frame_series(train_dfs)
        val_dataset = self.frame_series(val_dfs)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        return train_loader, val_loader, nb_features
