from torch.utils.data import Dataset


class JobDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.x = (df['name']+' '+df['description']).values
        self.y = df['target'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
