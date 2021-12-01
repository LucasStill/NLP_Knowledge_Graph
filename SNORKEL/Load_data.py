import pandas as pd
from itertools import chain


def get_tok2idx(data):
    vocab = list(set(data['Word'].to_list()))

    tok2idx = {tok: idx for idx, tok in enumerate(vocab)}

    return tok2idx


def load_data():
    # Load data
    df = pd.read_csv(r'C:\Users\JJ199\Downloads\kaggle_at_home.csv')

    # Create index for all words (Bag of Words)
    token2idx = get_tok2idx(df)

    # Add indices
    df['Word_idx'] = df['Word'].map(token2idx)

    # Fill na
    df['Sentence #'] = df['Sentence #'].fillna(method='ffill', axis=0)

    # Groupby and collect columns
    df = df.groupby(['Sentence #'],
                    as_index=False)['Word', 'Word_idx', 'Tag'].agg(lambda x: list(x))

    print(df.head().iloc[0])

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
