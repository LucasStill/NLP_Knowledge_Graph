import pandas as pd
from itertools import chain
import ast


def get_tok2idx(Words):
    vocab = list(set(Words))

    tok2idx = {tok: idx for idx, tok in enumerate(vocab)}

    return tok2idx

def create_token_list(words, token2idx):
    return [token2idx[word] for word in words]


def load_data():
    # Load data
    df = pd.read_csv(r'C:\Users\JJ199\Downloads\file_regex001.csv', nrows=50)

    # Set correct type for list
    df['Word'] = df['Word'].apply(ast.literal_eval)

    # Create index for all words (Bag of Words)
    token2idx = get_tok2idx(df['Word'].apply(pd.Series).stack().reset_index(drop = True))

    # Add indices
    df['Word_idx'] = df['Word'].map(lambda x: create_token_list(x, token2idx))

    # Groupby and collect columns
    df = df[['Word', 'Word_idx', 'Tag']]

    return df


def create_tag_df(df, label1: str, label2: str):
    # For every sentence
    ids1 = []
    ids2 = []
    for i in range(df.shape[0]):
        # Get PERSON_ID and tokens
        id1 = -1
        id2 = -1
        tags = df.iloc[i]['Tag']
        for j in range(len(tags)):
            if 'B {}'.format(label1) in tags[j]:
                if id1 == -1:
                    id1 = j
            elif 'B {}'.format(label2) in tags[j]:
                if id2 == -1:
                    id2 = j

        # Add tokens and
        ids1.append(id1 if id1 != -1 and id2 != -1 else -1)
        ids2.append(id2 if id1 != -1 and id2 != -1 else -1)

    # Create dataframe for snorkel
    df_train = pd.DataFrame({'tokens': df.Word, 'id1': ids1, 'id2': ids2,
                             'between_tokens': [[] for _ in range(df.shape[0])],
                             'text_left_1': [[] for _ in range(df.shape[0])],
                             'text_left_2': [[] for _ in range(df.shape[0])],
                             'text_right_1': [[] for _ in range(df.shape[0])],
                             'text_right_2': [[] for _ in range(df.shape[0])]})

    return df_train

def create_tag_df_special(df, label1: list, label2: list):
    # For every sentence
    ids1 = []
    ids2 = []
    for i in range(df.shape[0]):
        # Get PERSON_ID and tokens
        id1 = -1
        id2 = -1
        tags = df.iloc[i]['Tag']
        for j in range(len(tags)):
            if any('B {}'.format(x) in tags[j] for x in label1):
                if id1 == -1:
                    id1 = j
            elif any('B {}'.format(x) in tags[j] for x in label2):
                if id2 == -1:
                    id2 = j

        # Add tokens and
        ids1.append(id1 if id1 != -1 and id2 != -1 else -1)
        ids2.append(id2 if id1 != -1 and id2 != -1 else -1)

    # Create dataframe for snorkel
    df_train = pd.DataFrame({'tokens': df.Word, 'id1': ids1, 'id2': ids2,
                             'between_tokens': [[] for _ in range(df.shape[0])],
                             'text_left_1': [[] for _ in range(df.shape[0])],
                             'text_left_2': [[] for _ in range(df.shape[0])],
                             'text_right_1': [[] for _ in range(df.shape[0])],
                             'text_right_2': [[] for _ in range(df.shape[0])]})

    return df_train
