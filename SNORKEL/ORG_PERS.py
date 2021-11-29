import spacy
from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
import pandas as pd
from snorkel.labeling.model import LabelModel
from snorkel.preprocess import preprocessor

# PREPROCESSING STEPS
@preprocessor()
def get_text_between(cand):
    """
    Returns the text between the two person mentions in the sentence for a candidate
    """
    start = cand.id1 + 1
    end = cand.id2
    cand.between_tokens = cand.tokens[start:end]
    return cand

@preprocessor()
def get_text_left(cand):
    """
    Returns the text left to the two persons mentioned in the sentence for a candidate
    """
    cand.text_left_1 = cand.tokens[:cand.id1]
    cand.text_left_2 = cand.tokens[:cand.id2]
    return cand

@preprocessor()
def get_text_right(cand):
    """
    Returns the text left to the two persons mentioned in the sentence for a candidate
    """
    cand.text_right_1 = cand.tokens[(cand.id1 + 1):]
    cand.text_right_2 = cand.tokens[(cand.id2 + 1):]
    return cand

# Load spacy
nlp = spacy.load("en_core_web_sm")

# Create a dataset
sentences = ["The European Court of Human Rights (Third Section), sitting as a Chamber composed of:	Paul Lemmens, President, Georgios A. Serghides, Dmitry Dedov and Georges Ravarani", "This is a test", "The is an example"]

# For every sentence
tokens_df = []
ids1 = []
ids2 = []
for sentence in sentences:
    # Get PERSON_ID and date_id and tokens
    id1 = -1
    id2 = -1
    tokens = []
    for token in nlp(sentence):
        print(token, token.ent_type_)
        tokens.append(str(token))
        if token.ent_type_ == 'ORG':
            if id1 == -1:
                id1 = token.i
        if token.ent_type_ == 'PERSON':
            if id2 == -1:
                id2 = token.i

    # Add tokens and
    tokens_df.append(tokens)
    ids1.append(id1 if id1 != -1 and id2 != -1 else -1)
    ids2.append(id2 if id1 != -1 and id2 != -1 else -1)



# Create dataframe for snorkel
df_train = pd.DataFrame({'tokens':tokens_df, 'id1':ids1, 'id2':ids2,
                       'between_tokens':[[] for _ in range(len(tokens_df))], 'text_left_1':[[] for _ in range(len(tokens_df))],
                         'text_left_2':[[] for _ in range(len(tokens_df))], 'text_right_1':[[] for _ in range(len(tokens_df))],
                         'text_right_2':[[] for _ in range(len(tokens_df))]})

# Print table
print(df_train)

# Set Classes for relations (e.g. lawyer, bornin, etc.)
NEGATIVE = 0
COMPOSED = 1
MORE = 2

# Labeling Functions
# Capture constituent relation
composed = {"composed", "constituted", "comprised"}
@labeling_function(resources=dict(composed=composed), pre=[get_text_between])
def lf_composed(x, composed):
    return COMPOSED if len(composed.intersection(set(x.between_tokens))) > 0 else NEGATIVE

# Emotional relation
more = {"more"}
@labeling_function(resources=dict(more=more), pre=[get_text_between])
def lf_more(x, more):
    return MORE if len(more.intersection(set(x.between_tokens))) > 0 else NEGATIVE

@labeling_function()
def lf_2(x):
    return NEGATIVE

# Use all labeling functions
lfs = [
    lf_composed, lf_more, lf_2
]

# Apply them
applier = PandasLFApplier(lfs)
L_train = applier.apply(df_train)
print(L_train)

# Train labelingmodel
label_model = LabelModel(cardinality=3, verbose=True)
label_model.fit(L_train, n_epochs=5000, log_freq=500, seed=12345)

from snorkel.utils import probs_to_preds

probs_train = label_model.predict_proba(L_train)
preds_train = probs_to_preds(probs_train)

print(probs_train, preds_train)
