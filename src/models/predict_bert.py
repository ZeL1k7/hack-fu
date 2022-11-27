import itertools
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from src import load_job_model


if __name__ == "__main__":
    df_test = pd.read_csv('/mnt/disk_d/hack-fu/data/raw/test.csv')
    df_okz = pd.read_csv('/mnt/disk_d/hack-fu/data/raw/okz_3_4_professions.csv', sep='\t')
    submission = pd.read_csv('/mnt/disk_d/hack-fu/data/raw/sample.csv')
    inverted_targets_dict = dict(zip(np.arange(df_okz['code'].values.shape[0])), df_okz['code'].values)

    test_dataset = (df_test['name'] + ' ' + df_test['description']).astype(str).values.tolist()
    loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = load_job_model('/kaggle/input/state-dict-bert/state_dict.pt')
    device = model.device
    model.to(device)

    predictions = []
    with torch.no_grad():
        for samples in tqdm(loader):
            predictions.append(torch.argmax(F.softmax(model(samples), dim=1), dim=1).cpu().numpy())

    model = load_job_model('/kaggle/input/state-dict-bert/state_dict.pt')
    device = model.device
    model.to(device)

    submission_targets = np.array(list(itertools.chain.from_iterable(predictions)))
    submission['target'] = submission_targets
    submission['target'] = submission['target'].apply(lambda x: inverted_targets_dict[x])

    submission.to_csv('bert_baseline.csv', index=False)


