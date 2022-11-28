import pandas as pd
import os


# n-gram comparing words in sentences function
def ngram_words_compare(words1:list, words2:list):
    score = 0
    for i in range(1, len(words1) + 1):
        n = i
        # n-gram
        ngram1 = [words1[i:i+n] for i in range(len(words1)-n+1)]
        ngram2 = [words2[i:i+n] for i in range(len(words2)-n+1)]

        for ngram in ngram1:
            if ngram in ngram2:
                score += 1*len(ngram)
    return score


# find best match okz description for position name
def find_best_match(df_position_name) -> dict:
    position_name_okz = {}
    for index, row in df_position_name.iterrows():
        # os clean console
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f'{index}/{len(df_position_name)}')
        
        position_name = row['Position']
        position_name_okz[index] = {}
        for code in okz['code']:
            list_position_name = position_name.split()
            list_description = okz[okz['code'] == code]['description'].values[0].split()
            for word in list_position_name:
                if word in list_description:
                    score = ngram_words_compare(list_position_name, list_description)
                    if score > 0:
                        position_name_okz[index][code] = score
    return position_name_okz


if __name__ == '__main__':
    labeled = pd.read_csv('../../data/interim/labeled.csv')
    unlabeled = pd.read_csv('../../data/interim/unlabeled.csv')
    okz = pd.read_csv('../../data/interim/okz.csv', sep=',')

    labeled = labeled.iloc[:20]

    print('finding dicts')
    position_name_okz = find_best_match(labeled)

    print('finding best')
    position_name_okz_best = [max(position_name_okz[ind], key=position_name_okz[ind].get, default=-1) for ind in position_name_okz]

    labeled['target2'] = position_name_okz_best

    for i in range(len(labeled)):
        print(f"{labeled['Position'][i]} - {labeled['target'][i]} - {labeled['target2'][i]}")
    # find count target1 == target2
    count = 0
    for index, row in labeled.iterrows():
        if row['target'] == row['target2']:
            count += 1

    print(f"{count}/{len(labeled)}")