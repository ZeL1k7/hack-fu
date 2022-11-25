import re
import pandas as pd
from pathlib import Path


class MergeData:

    def __init__(self, raw_data_path: Path, output_data_path: Path):
        self.raw_data_path = raw_data_path
        self.output_data_path = output_data_path
        self.html_regexp = re.compile(r'<.*?>')
        self.df_train = pd.read_csv(raw_data_path+'train.csv')
        self.df_json = pd.read_json(raw_data_path + 'vacancy_descriptions/1_parsed.json')
        self.json_paths = [raw_data_path + 'vacancy_descriptions/' + str(i) + '_parsed.json' for i in range(2, 6)]

    def merge(self):
        for path in self.json_paths:
            df_json_temp = pd.read_json(path)
            self.df_json = pd.concat([self.df_json, df_json_temp], axis=0)

        self.df_json['content'] = self.df_json['Content'].apply(lambda x: list(x.values()))
        self.df_json['index'] = self.df_json['ID'].rename('index')
        self.df_json = self.df_json.drop(['Content', 'ID'],axis=1)

        df = self.df_json.merge(self.df_train, how='left', on='index')
        df = df.drop('name', axis=1)
        df['target'] = df['target'].fillna(-1.0)
        df['target'] = df['target'].astype(int)
        df['description'] = df['description'].astype(str)
        df['description'] = df['description'].apply(lambda x:  self.html_regexp.sub(r'', x))

        df_labeled = df[df['target'] != -1].to_csv(self.output_data_path+'labeled.csv', index=False)
        df_unlabeled = df[df['target'] == -1].to_csv(self.output_data_path+'unlabeled.csv', index=False)