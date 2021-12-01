from Load_data import load_data
from extract_relations import *

def combine_lists(list1, list2):
    return [list2[i] if list1[i] == "NEGATIVE" else list1[i] for i in range(len(list1))]

# Load dataframe
df = load_data()

# Load all relation extracting functions
relation_functions = [get_PERSON_PERSON, get_PERSON_DATE, get_PERSON_GPE, get_PERSON_MONEY,
                      get_ORG_PERSON, get_ORG_DATE, get_EVENT_DATE]

# Combine all outputs
for i, fn in enumerate(relation_functions):
    # Get all labels using Snorkel
    lbl = fn(df)

    # Add label to list, if it's still free
    if i == 0:
        label = lbl
    else:
        label = combine_lists(label, lbl)

# Add relation
df['Relation'] = label

# Save
df.to_pickle("Relations2.pkl")
