from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel
from snorkel.preprocess import preprocessor
from snorkel.utils import probs_to_preds
from Load_data import create_tag_df

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

def get_PERSON_PERSON(df):
    df_train = create_tag_df(df, "PERSON", "PERSON")

    # Set Classes for relations (e.g. lawyer, bornin, etc.)
    labels = {0:'NEGATIVE', 1:'LAWYER'}
    NEGATIVE = 0
    LAWYER = 1

    # Labeling Functions
    # Capture lawyer relation
    lawyer = {"represented"}
    @labeling_function(resources=dict(lawyer=lawyer), pre=[get_text_between])
    def lf_lawyer(x, lawyer):
        return LAWYER if len(lawyer.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Emotional relation
    @labeling_function()
    def lf_empty1(x):
        return NEGATIVE

    @labeling_function()
    def lf_empty2(x):
        return NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_lawyer, lf_empty1, lf_empty2
    ]

    # Apply them
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df_train)

    # Train labelingmodel
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train, n_epochs=5000, log_freq=500, seed=12345)

    probs_train = label_model.predict_proba(L_train)
    preds_train = probs_to_preds(probs_train)

    return list(map(lambda x: labels[x], preds_train))

def get_PERSON_DATE(df):
    df_train = create_tag_df(df,"PERSON", "DATE")

    # Set Classes for relations (e.g. lawyer, bornin, etc.)
    labels = {0:'NEGATIVE', 1:'BORNIN', 2:'SENTENCED'}
    NEGATIVE = 0
    BORNIN = 1
    SENTENCED = 2

    # Labeling Functions
    # Capture birth relation
    born = {"born", "birthdate", "birth-date"}
    @labeling_function(resources=dict(born=born), pre=[get_text_between])
    def lf_born(x, born):
        return BORNIN if len(born.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Sentence/conviction relation
    sentenced = {"sentenced", "judgement", "convict", "convicted", "condemned", "punished", "penalized"}
    @labeling_function(resources=dict(sentenced=sentenced), pre=[get_text_between])
    def lf_sentenced(x, sentenced):
        return SENTENCED if len(sentenced.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    @labeling_function()
    def lf_empty(x):
        return NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_born, lf_sentenced, lf_empty
    ]

    # Apply them
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df_train)

    # Train labelingmodel
    label_model = LabelModel(cardinality=3, verbose=True)
    label_model.fit(L_train, n_epochs=5000, log_freq=500, seed=12345)

    # Get predictions
    probs_train = label_model.predict_proba(L_train)
    preds_train = probs_to_preds(probs_train)

    return list(map(lambda x: labels[x], preds_train))

def get_PERSON_GPE(df):
    df_train = create_tag_df(df, "PERSON", "GPE")

    # Set Classes for relations (e.g. lawyer, bornin, etc.)
    labels = {0:'NEGATIVE', 1:'BORNIN', 2:'RESIDENT', 3:'CITIZEN'}
    NEGATIVE = 0
    BORNIN = 1
    RESIDENT = 2
    CITIZEN = 3

    # Labeling Functions
    # Capture lawyer relation
    born = {"born", "took birth in"}
    @labeling_function(resources=dict(born=born), pre=[get_text_between])
    def lf_born(x, born):
        return BORNIN if len(born.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Emotional relation
    resident = {"resides", "resided", "residing", "lived", "lives", "living"}
    @labeling_function(resources=dict(resident=resident), pre=[get_text_between])
    def lf_resident(x, resident):
        return RESIDENT if len(resident.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    citizen = {"citizen", "national", "native", "subject"}
    @labeling_function(resources=dict(citizen=citizen), pre=[get_text_between])
    def lf_citizen(x, citizen):
        return CITIZEN if len(citizen.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_born, lf_resident, lf_citizen
    ]

    # Apply them
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df_train)

    # Train labelingmodel
    label_model = LabelModel(cardinality=4, verbose=True)
    label_model.fit(L_train, n_epochs=5000, log_freq=500, seed=12345)

    probs_train = label_model.predict_proba(L_train)
    preds_train = probs_to_preds(probs_train)

    return list(map(lambda x: labels[x], preds_train))

def get_PERSON_MONEY(df):
    df_train = create_tag_df(df, "PERSON", "MONEY")

    # Set Classes for relations (e.g. lawyer, bornin, etc.)
    labels = {0:'NEGATIVE', 1:'CLAIM'}
    NEGATIVE = 0
    CLAIM = 1

    # Labeling Functions
    # Capture claimed amount relation
    claim = {"claimed", "received"}
    @labeling_function(resources=dict(claim=claim), pre=[get_text_between])
    def lf_claim(x, claim):
        return CLAIM if len(claim.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    @labeling_function()
    def lf_empty1(x):
        return NEGATIVE

    @labeling_function()
    def lf_empty2(x):
        return NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_claim, lf_empty1, lf_empty2
    ]

    # Apply them
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df_train)

    # Train labelingmodel
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train, n_epochs=5000, log_freq=500, seed=12345)

    probs_train = label_model.predict_proba(L_train)
    preds_train = probs_to_preds(probs_train)

    return list(map(lambda x: labels[x], preds_train))

def get_ORG_PERSON(df):
    df_train = create_tag_df(df, "ORG", "PERSON")

    # Set Classes for relations (e.g. lawyer, bornin, etc.)
    labels = {0:'NEGATIVE', 1:'COMPOSED'}
    NEGATIVE = 0
    COMPOSED = 1

    # Labeling Functions
    # Capture constituent relation
    composed = {"composed", "constituted", "comprised"}
    @labeling_function(resources=dict(composed=composed), pre=[get_text_between])
    def lf_composed(x, composed):
        return COMPOSED if len(composed.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    @labeling_function()
    def lf_empty1(x):
        return NEGATIVE

    @labeling_function()
    def lf_empty2(x):
        return NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_composed, lf_empty1, lf_empty2
    ]

    # Apply them
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df_train)

    # Train labelingmodel
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train, n_epochs=5000, log_freq=500, seed=12345)

    probs_train = label_model.predict_proba(L_train)
    preds_train = probs_to_preds(probs_train)

    return list(map(lambda x: labels[x], preds_train))

def get_ORG_DATE(df):
    df_train = create_tag_df(df, "ORG", "DATE")

    # Set Classes for relations (e.g. lawyer, bornin, etc.)
    labels = {0:'NEGATIVE', 1:'BORNIN', 2:'DELIBERATED', 3:'ORIGINATES'}
    NEGATIVE = 0
    BORNIN = 1
    DELIBERATED = 2
    ORIGINATES = 3

    # Labeling Functions
    # Capture lawyer relation
    born = {"born", "birthday", "birthdata", "birth-date"}
    @labeling_function(resources=dict(born=born), pre=[get_text_between])
    def lf_born(x, born):
        return BORNIN if len(born.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Emotional relation
    deliberated = {"deliberated", "deliberate", "studied", "considered"}
    @labeling_function(resources=dict(deliberated=deliberated), pre=[get_text_between])
    def lf_deliberated(x, deliberated):
        return DELIBERATED if len(deliberated.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    originates = {"originates", "originate"}
    @labeling_function(resources=dict(originates=originates), pre=[get_text_between])
    def lf_originates(x, originates):
        return ORIGINATES if len(originates.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_born, lf_deliberated, lf_originates
    ]

    # Apply them
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df_train)

    # Train labelingmodel
    label_model = LabelModel(cardinality=4, verbose=True)
    label_model.fit(L_train, n_epochs=5000, log_freq=500, seed=12345)

    probs_train = label_model.predict_proba(L_train)
    preds_train = probs_to_preds(probs_train)

    return list(map(lambda x: labels[x], preds_train))

def get_EVENT_DATE(df):
    df_train = create_tag_df(df, "EVENT", "DATE")

    # Set Classes for relations (e.g. lawyer, bornin, etc.)
    labels = {0:'NEGATIVE', 1:'HEARING'}
    NEGATIVE = 0
    HEARING = 1

    # Labeling Functions
    # Capture hearing date relation
    hearing = {"held on", "open on"}
    @labeling_function(resources=dict(hearing=hearing), pre=[get_text_between])
    def lf_hearing(x, hearing):
        return HEARING if len(hearing.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    @labeling_function()
    def lf_empty1(x):
        return NEGATIVE

    @labeling_function()
    def lf_empty2(x):
        return NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_hearing, lf_empty1, lf_empty2
    ]

    # Apply them
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df_train)

    # Train labelingmodel
    label_model = LabelModel(cardinality=3, verbose=True)
    label_model.fit(L_train, n_epochs=5000, log_freq=500, seed=12345)

    probs_train = label_model.predict_proba(L_train)
    preds_train = probs_to_preds(probs_train)

    return list(map(lambda x: labels[x], preds_train))

