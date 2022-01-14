from Load_data import load_data
from extract_relations import *

def combine_lists(list1, list2):
    return [list2[i] if list1[i] == "NEGATIVE" else list1[i] for i in range(len(list1))]

# Load dataframe
df = load_data()

# Load all relation extracting functions
relation_functions = [get_DATE_COURT, get_PERSON_PERSON, get_PERSON_DATE, get_PERSON_GPE,
                      get_PERSON_MONEY, get_PERSON_ARTICLE, get_PERSON_RULE, get_PERSON_NORP,
                      get_PERSON_ORG, get_PERSON_COURT, get_ORG_PERSON, get_ORG_DATE, get_ORG_ARTICLE,
                      get_ORG_RULE, get_COURT_PERSON, get_COURT_ARTICLE, get_COURT_PROTOCOL,
                      get_COURT_RULE, get_COURT_DATE, get_NORP_RULE, get_NORP_ARTICLE, get_NORP_PERSON,
                      get_APPLICATION_GPE, get_APPLICATION_PERSON, get_APPLICATION_DATE,
                      get_APPLICATION_ARTICLE, get_CASE_ORG, get_CASE_DATE, get_CASE_COURT,
                      get_CASE_PERSON, get_CASE_APPLICATION, get_PROSECUTOR_PERSON, get_PROSECUTOR_ORG,
                      get_PROSECUTOR_NORP, get_PROSECUTOR_COURT, get_PROSECUTOR_DATE
                      ]

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
df.to_csv("annotated_all.csv", sep="|")
