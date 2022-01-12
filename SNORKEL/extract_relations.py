from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel
from snorkel.preprocess import preprocessor
from snorkel.utils import probs_to_preds
from Load_data import create_tag_df, create_tag_df_special

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

def get_DATE_COURT(df):
    #Example: "On 16 March 2006 the Erzurum Assize Court acquitted the applicant"
    #Example: "On 17 April 1998 the Constitutional Court quashed the decision of NJC of 14 January 1997"

    df_train = create_tag_df(df, "DATE", "COURT")

    # Set Classes for relations (e.g. acquitted etc.)
    labels = {0: 'NEGATIVE', 1: 'QUASHEDJUDGMENT', 2: 'ACQUITTED', 3: 'QUASHEDDECISION'}
    NEGATIVE = 0
    QUASHEDJUDGMENT = 1
    ACQUITTED = 2
    QUASHEDDECISION = 3

    # Labeling Functions
    # Capture "quashed judgment" relation to the right of DATE and COURT entity
    quashedjudgment = {"quashed the judgment", "quashed judgment", "judgment was quashed"}
    @labeling_function(resources=dict(quashedjudgment=quashedjudgment), pre=[get_text_right])
    def lf_quashedjudgment(x, quashedjudgment):
        return QUASHEDJUDGMENT if len(quashedjudgment.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "acquitted" relation to the right of DATE and COURT entity
    acquitted = {"acquitted"}
    @labeling_function(resources=dict(acquitted=acquitted), pre=[get_text_right])
    def lf_acquitted(x, acquitted):
        return ACQUITTED if len(acquitted.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "quashed decision" relation to the right of DATE and COURT entity
    quasheddecision = {"quashed the decision", "quashed decision", "decision was quashed"}
    @labeling_function(resources=dict(quasheddecision=quasheddecision), pre=[get_text_right])
    def lf_quasheddecision(x, quasheddecision):
        return QUASHEDDECISION if len(quasheddecision.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_quashedjudgment, lf_acquitted, lf_quasheddecision
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

def get_PERSON_PERSON(df):
    # Example = "Mr Krunislav Olujić  was represented by Mr B. Hajduković"
    df_train = create_tag_df_special(df, ['JUDGE', 'REGISTRAR', 'SECRETARY', 'LAWYER', 'DEFENDANT', 'INVESTIGATORS', 'PROSECUTOR', 'PERSON'],
                                     ['JUDGE', 'REGISTRAR', 'SECRETARY', 'LAWYER', 'DEFENDANT', 'INVESTIGATORS', 'PROSECUTOR', 'PERSON'])

    # Set Classes for relations (e.g. lawyer etc.)
    labels = {0: 'NEGATIVE', 1: 'LAWYER', 2: 'ACCUSED', 3: 'ALLEGED', 4: 'AUTHORISED'}
    NEGATIVE = 0
    LAWYER = 1
    ACCUSED = 2
    ALLEGED = 3
    AUTHORISED = 4

    # Labeling Functions
    # Capture lawyer relation
    lawyer = {"represented"}
    @labeling_function(resources=dict(lawyer=lawyer), pre=[get_text_between])
    def lf_lawyer(x, lawyer):
        return LAWYER if len(lawyer.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "accused" relation between PERSON and PERSON
    accused = {"accused"}
    @labeling_function(resources=dict(accused=accused), pre=[get_text_between])
    def lf_accused(x, accused):
        return ACCUSED if len(accused.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "alleged" relation between PERSON and PERSON
    alleged = {"alleged"}
    @labeling_function(resources=dict(alleged=alleged), pre=[get_text_between])
    def lf_alleged(x, alleged):
        return ALLEGED if len(alleged.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "authorized" relation between PERSON and PERSON
    authorised = {"authorized", "authorised"}
    @labeling_function(resources=dict(authorised=authorised), pre=[get_text_between])
    def lf_authorised(x, authorised):
        return AUTHORISED if len(authorised.intersection(set(x.between_tokens))) > 0 else NEGATIVE


    # Use all labeling functions
    lfs = [
        lf_lawyer, lf_accused, lf_alleged, lf_authorised
    ]

    # Apply them
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df_train)

    # Train labelingmodel
    label_model = LabelModel(cardinality=5, verbose=True)
    label_model.fit(L_train, n_epochs=5000, log_freq=500, seed=12345)

    probs_train = label_model.predict_proba(L_train)
    preds_train = probs_to_preds(probs_train)

    return list(map(lambda x: labels[x], preds_train))

def get_PERSON_DATE(df):
    # Example = "Dr. Albert appealed to the French-language Appeals Council of the Ordre on 18 June"

    df_train = create_tag_df(df, ['JUDGE', 'REGISTRAR', 'SECRETARY', 'LAWYER', 'DEFENDANT', 'INVESTIGATORS', 'PROSECUTOR', 'PERSON'],
                             ["DATE"])

    # Set Classes for relations (e.g. bornin, etc.)
    labels = {0: 'NEGATIVE', 1: 'BORNIN', 2: 'SENTENCED', 3: 'APPEALEDON'}
    NEGATIVE = 0
    BORNIN = 1
    SENTENCED = 2
    APPEALEDON = 3

    # Labeling Functions
    # Capture "birth" relation between Person and DATE
    born = {"born", "birthdate", "birth-date"}
    @labeling_function(resources=dict(born=born), pre=[get_text_between])
    def lf_born(x, born):
        return BORNIN if len(born.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "Sentence/conviction" relation between Person and DATE
    sentenced = {"sentenced", "judgement", "convict", "convicted", "condemned", "punished", "penalized"}
    @labeling_function(resources=dict(sentenced=sentenced), pre=[get_text_between])
    def lf_sentenced(x, sentenced):
        return SENTENCED if len(sentenced.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "appealed" relation between Person and DATE
    appealed = {"appealed"}
    @labeling_function(resources=dict(appealed=appealed), pre=[get_text_between])
    def lf_appealed(x, appealed):
        return APPEALEDON if len(appealed.intersection(set(x.between_tokens))) > 0 else NEGATIVE

        # Use all labeling functions

    lfs = [
        lf_born, lf_sentenced, lf_appealed
    ]

    # Apply them
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df_train)

    # Train labelingmodel
    label_model = LabelModel(cardinality=4, verbose=True)
    label_model.fit(L_train, n_epochs=5000, log_freq=500, seed=12345)

    # Get predictions
    probs_train = label_model.predict_proba(L_train)
    preds_train = probs_to_preds(probs_train)

    return list(map(lambda x: labels[x], preds_train))

def get_PERSON_GPE(df):
    df_train = create_tag_df(df, ['JUDGE', 'REGISTRAR', 'SECRETARY', 'LAWYER', 'DEFENDANT', 'INVESTIGATORS', 'PROSECUTOR', 'PERSON'],
                             ["GPE"])

    # Set Classes for relations (e.g. bornin, resident etc.)
    labels = {0: 'NEGATIVE', 1: 'BORNIN', 2: 'RESIDENT', 3: 'CITIZEN', 4: 'PRACTISINGIN', 5: 'DETAINEDIN'}
    NEGATIVE = 0
    BORNIN = 1
    RESIDENT = 2
    CITIZEN = 3
    PRACTISINGIN = 4
    DETAINEDIN = 5

    # Labeling Functions
    # Capture lawyer relation
    born = {"born", "took birth in"}
    @labeling_function(resources=dict(born=born), pre=[get_text_between])
    def lf_born(x, born):
        return BORNIN if len(born.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Residence relation
    resident = {"resides", "resided", "residing", "lived", "lives", "living"}
    @labeling_function(resources=dict(resident=resident), pre=[get_text_between])
    def lf_resident(x, resident):
        return RESIDENT if len(resident.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    citizen = {"citizen", "national", "native", "subject"}
    @labeling_function(resources=dict(citizen=citizen), pre=[get_text_between])
    def lf_citizen(x, citizen):
        return CITIZEN if len(citizen.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "practising in" relation between Person and GPE
    practising = {"practising"}
    @labeling_function(resources=dict(practising=practising), pre=[get_text_between])
    def lf_practising(x, practising):
        return PRACTISINGIN if len(practising.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "detained in" relation between Person and GPE
    detained = {"detained"}
    @labeling_function(resources=dict(detained=detained), pre=[get_text_between])
    def lf_detained(x, detained):
        return DETAINEDIN if len(detained.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_born, lf_resident, lf_citizen, lf_practising, lf_detained
    ]

    # Apply them
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df_train)

    # Train labelingmodel
    label_model = LabelModel(cardinality=6, verbose=True)
    label_model.fit(L_train, n_epochs=5000, log_freq=500, seed=12345)

    probs_train = label_model.predict_proba(L_train)
    preds_train = probs_to_preds(probs_train)

    return list(map(lambda x: labels[x], preds_train))

def get_PERSON_MONEY(df):
    # Example = 'The applicant sought reparation for the pecuniary damage she had sustained, which she put at 66,864,000 Italian lire (ITL)'
    df_train = create_tag_df(df, ['JUDGE', 'REGISTRAR', 'SECRETARY', 'LAWYER', 'DEFENDANT', 'INVESTIGATORS', 'PROSECUTOR', 'PERSON'],
                             ["MONEY"])

    # Set Classes for relations (e.g. sought reparation, claimed etc.)
    labels = {0: 'NEGATIVE', 1: 'CLAIM', 2: 'RECEIVED', 3: 'SOUGHTREPARATION', 4: 'SOUGHTREIMBURSEMENT'}
    NEGATIVE = 0
    CLAIM = 1
    RECEIVED = 2
    SOUGHTREPARATION = 3
    SOUGHTREIMBURSEMENT = 4

    # Labeling Functions
    # Capture "claimed amount" relation
    claim = {"claimed"}
    @labeling_function(resources=dict(claim=claim), pre=[get_text_between])
    def lf_claim(x, claim):
        return CLAIM if len(claim.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "received amount" relation
    received = {"received"}
    @labeling_function(resources=dict(received=received), pre=[get_text_between])
    def lf_received(x, received):
        return RECEIVED if len(received.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "reparation amount" relation
    reparation = {"sought reparation"}
    @labeling_function(resources=dict(reparation=reparation), pre=[get_text_between])
    def lf_reparation(x, reparation):
        return SOUGHTREPARATION if len(reparation.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "reimbursement amount" relation
    reimbursement = {"sought reimbursement"}
    @labeling_function(resources=dict(reimbursement=reimbursement), pre=[get_text_between])
    def lf_reimbursement(x, reimbursement):
        return SOUGHTREIMBURSEMENT if len(reimbursement.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_claim, lf_received, lf_reparation, lf_reimbursement
    ]

    # Apply them
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df_train)

    # Train labelingmodel
    label_model = LabelModel(cardinality=5, verbose=True)
    label_model.fit(L_train, n_epochs=5000, log_freq=500, seed=12345)

    probs_train = label_model.predict_proba(L_train)
    preds_train = probs_to_preds(probs_train)

    return list(map(lambda x: labels[x], preds_train))

def get_PERSON_ARTICLE(df):
    # Example = "The applicant complained under Article 1 of Protocol No. 1 "

    df_train = create_tag_df(df, ['JUDGE', 'REGISTRAR', 'SECRETARY', 'LAWYER', 'DEFENDANT', 'INVESTIGATORS', 'PROSECUTOR', 'PERSON'],
                             ["ARTICLE"])

    # Set Classes for relations (e.g. violated etc.)
    labels = {0: 'NEGATIVE', 1: 'VIOLATED', 2: 'BREACHED', 3: 'COMPLAINEDUNDER', 4: 'ALLEGEDBREACH'}
    NEGATIVE = 0
    VIOLATED = 1
    BREACHED = 2
    COMPLAINEDUNDER = 3
    ALLEGEDBREACH = 4

    # Labeling Functions
    # Capture "violated" relation between PERSON and ARTICLE
    violated = {"violated"}
    @labeling_function(resources=dict(violated=violated), pre=[get_text_between])
    def lf_violated(x, violated):
        return VIOLATED if len(violated.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "breached" relation between PERSON and ARTICLE
    breached = {"breached", "breach"}
    @labeling_function(resources=dict(breached=breached), pre=[get_text_between])
    def lf_breached(x, breached):
        return BREACHED if len(breached.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "complained under" relation between PERSON and ARTICLE
    complained = {"complained under"}
    @labeling_function(resources=dict(complained=complained), pre=[get_text_between])
    def lf_complained(x, complained):
        return COMPLAINEDUNDER if len(complained.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "alleged breach" relation between PERSON and ARTICLE
    alleged = {"alleged breach", "alleged a breach"}
    @labeling_function(resources=dict(alleged=alleged), pre=[get_text_between])
    def lf_alleged(x, alleged):
        return ALLEGEDBREACH if len(alleged.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_violated, lf_breached, lf_complained, lf_alleged
    ]

    # Apply them
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df_train)

    # Train labelingmodel
    label_model = LabelModel(cardinality=5, verbose=True)
    label_model.fit(L_train, n_epochs=5000, log_freq=500, seed=12345)

    probs_train = label_model.predict_proba(L_train)
    preds_train = probs_to_preds(probs_train)

    return list(map(lambda x: labels[x], preds_train))

def get_PERSON_RULE(df):
    # Example = "He was granted leave to represent himself before the Court, in accordance with Rule36 §§ 2 and 4 of the Rules of Court."

    df_train = create_tag_df(df, ['JUDGE', 'REGISTRAR', 'SECRETARY', 'LAWYER', 'DEFENDANT', 'INVESTIGATORS', 'PROSECUTOR', 'PERSON'],
                             ["RULE"])

    # Set Classes for relations (e.g. violated etc.)
    labels = {0: 'NEGATIVE', 1: 'VIOLATED', 2: 'GRANTEDLEAVE', 3: 'COMPLAINEDUNDER', 4: 'ALLEGEDBREACH'}
    NEGATIVE = 0
    VIOLATED = 1
    GRANTEDLEAVE = 2
    COMPLAINEDUNDER = 3
    ALLEGEDBREACH = 4

    # Labeling Functions
    # Capture "violated" relation between PERSON and RULE
    violated = {"violated"}
    @labeling_function(resources=dict(violated=violated), pre=[get_text_between])
    def lf_violated(x, violated):
        return VIOLATED if len(violated.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "granted leave" relation between PERSON and RULE
    grantleave = {"granted leave"}
    @labeling_function(resources=dict(grantleave=grantleave), pre=[get_text_between])
    def lf_grantleave(x, grantleave):
        return GRANTEDLEAVE if len(grantleave.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "complained under" relation between PERSON and RULE
    complained = {"complained under"}
    @labeling_function(resources=dict(complained=complained), pre=[get_text_between])
    def lf_complained(x, complained):
        return COMPLAINEDUNDER if len(complained.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "alleged breach" relation between PERSON and RULE
    alleged = {"alleged breach", "alleged a breach"}
    @labeling_function(resources=dict(alleged=alleged), pre=[get_text_between])
    def lf_alleged(x, alleged):
        return ALLEGEDBREACH if len(alleged.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_violated, lf_grantleave, lf_complained, lf_alleged
    ]

    # Apply them
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df_train)

    # Train labelingmodel
    label_model = LabelModel(cardinality=5, verbose=True)
    label_model.fit(L_train, n_epochs=5000, log_freq=500, seed=12345)

    probs_train = label_model.predict_proba(L_train)
    preds_train = probs_to_preds(probs_train)

    return list(map(lambda x: labels[x], preds_train))

def get_PERSON_NORP(df):
    # Example = "Dr. Alfred Albert is a medical practitioner. He was born in 1908, lives at Molenbeek and is a Belgian national"

    df_train = create_tag_df(df, ['JUDGE', 'REGISTRAR', 'SECRETARY', 'LAWYER', 'DEFENDANT', 'INVESTIGATORS', 'PROSECUTOR', 'PERSON'],
                             ["NORP"])

    # Set Classes for relations (e.g. "is" etc.)
    labels = {0: 'NEGATIVE', 1: 'LODGEDCOMPLAINTWITH', 2: 'APPEALEDTO', 3: 'LODGEDCOMPLAINTAGAINST',
              4: 'APPEALEDAGAINST'}
    NEGATIVE = 0
    LODGEDCOMPLAINTWITH = 1
    APPEALEDTO = 2
    LODGEDCOMPLAINTAGAINST = 3
    APPEALEDAGAINST = 4

    # Labeling Functions
    # Capture "lodged complaint with" relation between PERSON and ORG
    lodgedcomplaintwith = {"lodged a complaint with", "lodged complaint with", "complaint was lodged with"}
    @labeling_function(resources=dict(lodgedcomplaintwith=lodgedcomplaintwith), pre=[get_text_between])
    def lf_lodgedcomplaintwith(x, lodgedcomplaintwith):
        return LODGEDCOMPLAINTWITH if len(lodgedcomplaintwith.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "appealed to" relation between Person and ORG
    appealedto = {"appealed to"}
    @labeling_function(resources=dict(appealedto=appealedto), pre=[get_text_between])
    def lf_appealedto(x, appealedto):
        return APPEALEDTO if len(appealedto.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "lodged complaint against" relation between PERSON and ORG
    lodgedcomplaintagainst = {"lodged a complaint against", "lodged complaint against", "complaint was lodged against"}
    @labeling_function(resources=dict(lodgedcomplaintagainst=lodgedcomplaintagainst), pre=[get_text_between])
    def lf_lodgedcomplaintagainst(x, lodgedcomplaintagainst):
        return LODGEDCOMPLAINTAGAINST if len(
            lodgedcomplaintagainst.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "appealed against" relation between Person and ORG
    appealedagainst = {"appealed against"}
    @labeling_function(resources=dict(appealedagainst=appealedagainst), pre=[get_text_between])
    def lf_appealedagainst(x, appealedagainst):
        return APPEALEDAGAINST if len(appealedagainst.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_lodgedcomplaintwith, lf_appealedto, lf_lodgedcomplaintagainst, lf_appealedagainst
    ]

    # Apply them
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df_train)

    # Train labelingmodel
    label_model = LabelModel(cardinality=5, verbose=True)
    label_model.fit(L_train, n_epochs=5000, log_freq=500, seed=12345)

    probs_train = label_model.predict_proba(L_train)
    preds_train = probs_to_preds(probs_train)

    return list(map(lambda x: labels[x], preds_train))

def get_PERSON_ORG(df):
    # Example = "Dr. Albert appealed to the French-language Appeals Council of the Ordre on 18 June"

    df_train = create_tag_df(df, ['JUDGE', 'REGISTRAR', 'SECRETARY', 'LAWYER', 'DEFENDANT', 'INVESTIGATORS', 'PROSECUTOR', 'PERSON'],
                             ["ORG"])

    # Set Classes for relations (e.g. lodgedcomplaint etc.)
    labels = {0: 'NEGATIVE', 1: 'LODGEDCOMPLAINTWITH', 2: 'APPEALEDTO', 3: 'LODGEDCOMPLAINTAGAINST',
              4: 'APPEALEDAGAINST'}
    NEGATIVE = 0
    LODGEDCOMPLAINTWITH = 1
    APPEALEDTO = 2
    LODGEDCOMPLAINTAGAINST = 3
    APPEALEDAGAINST = 4

    # Labeling Functions
    # Capture "lodged complaint with" relation between PERSON and ORG
    lodgedcomplaintwith = {"lodged a complaint with", "lodged complaint with", "complaint was lodged with"}
    @labeling_function(resources=dict(lodgedcomplaintwith=lodgedcomplaintwith), pre=[get_text_between])
    def lf_lodgedcomplaintwith(x, lodgedcomplaintwith):
        return LODGEDCOMPLAINTWITH if len(lodgedcomplaintwith.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "appealed to" relation between Person and ORG
    appealedto = {"appealed to"}
    @labeling_function(resources=dict(appealedto=appealedto), pre=[get_text_between])
    def lf_appealedto(x, appealedto):
        return APPEALEDTO if len(appealedto.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "lodged complaint against" relation between PERSON and ORG
    lodgedcomplaintagainst = {"lodged a complaint against", "lodged complaint against", "complaint was lodged against"}
    @labeling_function(resources=dict(lodgedcomplaintagainst=lodgedcomplaintagainst), pre=[get_text_between])
    def lf_lodgedcomplaintagainst(x, lodgedcomplaintagainst):
        return LODGEDCOMPLAINTAGAINST if len(
            lodgedcomplaintagainst.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "appealed against" relation between Person and ORG
    appealedagainst = {"appealed against"}
    @labeling_function(resources=dict(appealedagainst=appealedagainst), pre=[get_text_between])
    def lf_appealedagainst(x, appealedagainst):
        return APPEALEDAGAINST if len(appealedagainst.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_lodgedcomplaintwith, lf_appealedto, lf_lodgedcomplaintagainst, lf_appealedagainst
    ]

    # Apply them
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df_train)

    # Train labelingmodel
    label_model = LabelModel(cardinality=5, verbose=True)
    label_model.fit(L_train, n_epochs=5000, log_freq=500, seed=12345)

    probs_train = label_model.predict_proba(L_train)
    preds_train = probs_to_preds(probs_train)

    return list(map(lambda x: labels[x], preds_train))

def get_PERSON_COURT(df):
    # Example= ""X and Y appealed to the Court of Appeal which, on 15 December 1977, referred the question to the Court of Justice of the European Communities"
    df_train = create_tag_df(df, ['JUDGE', 'REGISTRAR', 'SECRETARY', 'LAWYER', 'DEFENDANT', 'INVESTIGATORS', 'PROSECUTOR', 'PERSON'],
                             ["COURT"])

    # Set Classes for relations (e.g. appealed, informed, challenged etc.)
    labels = {0: 'NEGATIVE', 1: 'APPEALEDTO', 2: 'INFORMED', 3: 'CHALLENGEDDECISION', 4: 'SUBMITTED'}
    NEGATIVE = 0
    APPEALEDTO = 1
    INFORMED = 2
    CHALLENGEDDECISION = 3
    SUBMITTED = 4

    # Labeling Functions
    # Capture "appealed" relation between Person and Court
    appealed = {"appealed"}
    @labeling_function(resources=dict(appealed=appealed), pre=[get_text_between])
    def lf_appealed(x, appealed):
        return APPEALEDTO if len(appealed.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "informed" relation between Person and Court
    informed = {"informed"}
    @labeling_function(resources=dict(informed=informed), pre=[get_text_between])
    def lf_inform(x, informed):
        return INFORMED if len(informed.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "challenged decision" relation between Person and Court
    challenged = {"challenged"}
    @labeling_function(resources=dict(challenged=challenged), pre=[get_text_between])
    def lf_challengeddecision(x, challenged):
        return CHALLENGEDDECISION if len(challenged.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "submitted" relation between Person and Court
    submitted = {"submitted"}
    @labeling_function(resources=dict(submitted=submitted), pre=[get_text_between])
    def lf_submitted(x, submitted):
        return SUBMITTED if len(submitted.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_appealed, lf_inform, lf_challengeddecision, lf_submitted
    ]

    # Apply them
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df_train)

    # Train labelingmodel
    label_model = LabelModel(cardinality=5, verbose=True)
    label_model.fit(L_train, n_epochs=5000, log_freq=500, seed=12345)

    probs_train = label_model.predict_proba(L_train)
    preds_train = probs_to_preds(probs_train)

    return list(map(lambda x: labels[x], preds_train))

def get_ORG_PERSON(df):
    # Example = The Government appointed Mr. G. Raimondi as ad hoc judge to sit in his place (Article 27 § 2 of the Convention and Rule 29 § 2).

    df_train = create_tag_df(df, ["ORG"],
                             ['JUDGE', 'REGISTRAR', 'SECRETARY', 'LAWYER', 'DEFENDANT', 'INVESTIGATORS', 'PROSECUTOR', 'PERSON'])

    # Set Classes for relations (e.g. composed of, represented by etc.)
    labels = {0: 'NEGATIVE', 1: 'COMPOSED', 2: 'REPRESENTEDBY', 3: 'APPOINTED', 4:'INFORMED'}
    NEGATIVE = 0
    COMPOSED = 1
    REPRESENTEDBY = 2
    APPOINTED = 3
    INFORMED = 4

    # Labeling Functions
    # Capture "Chamber composition" relation between ORG and Person
    composed = {"composed", "constituted", "comprised"}
    @labeling_function(resources=dict(composed=composed), pre=[get_text_between])
    def lf_composed(x, composed):
        return COMPOSED if len(composed.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "represented by" relation between ORG and Person
    represented = {"represented by"}
    @labeling_function(resources=dict(represented=represented), pre=[get_text_between])
    def lf_represented(x, represented):
        return REPRESENTEDBY if len(represented.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "appointed" relation between ORG and Person
    appointed = {"appointed"}
    @labeling_function(resources=dict(appointed=appointed), pre=[get_text_between])
    def lf_appointed(x, appointed):
        return APPOINTED if len(appointed.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "informed" relation between ORG and Person
    informed = {"informed"}
    @labeling_function(resources=dict(informed=informed), pre=[get_text_between])
    def lf_informed(x, informed):
        return INFORMED if len(informed.intersection(set(x.between_tokens))) > 0 else NEGATIVE



    # Use all labeling functions
    lfs = [
        lf_composed, lf_represented, lf_appointed, lf_informed
    ]

    # Apply them
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df_train)

    # Train labelingmodel
    label_model = LabelModel(cardinality=5, verbose=True)
    label_model.fit(L_train, n_epochs=5000, log_freq=500, seed=12345)

    probs_train = label_model.predict_proba(L_train)
    preds_train = probs_to_preds(probs_train)

    return list(map(lambda x: labels[x], preds_train))

def get_ORG_DATE(df):
    df_train = create_tag_df(df, "ORG", "DATE")

    # Set Classes for relations (e.g. deliberated etc.)
    labels = {0: 'NEGATIVE', 1: 'BORNIN', 2: 'DELIBERATED', 3: 'ORIGINATES'}
    NEGATIVE = 0
    BORNIN = 1
    DELIBERATED = 2
    ORIGINATES = 3

    # Labeling Functions
    # Capture "born in" relation between Organization and Date
    born = {"born", "birthday", "birthdata", "birth-date"}
    @labeling_function(resources=dict(born=born), pre=[get_text_between])
    def lf_born(x, born):
        return BORNIN if len(born.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "deliberated" relation between Organization and Date
    deliberated = {"deliberated", "deliberate", "studied", "considered"}
    @labeling_function(resources=dict(deliberated=deliberated), pre=[get_text_between])
    def lf_deliberated(x, deliberated):
        return DELIBERATED if len(deliberated.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "originate" relation between Organization and Date
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

def get_ORG_ARTICLE(df):
    # Example = "The Government appointed Mr. G. Raimondi as ad hoc judge to sit in his place (Article 27 § 2 of the Convention and Rule 29 § 2)."

    df_train = create_tag_df(df, "ORG", "ARTICLE")

    # Set Classes for relations (e.g. violated etc.)
    labels = {0: 'NEGATIVE', 1: 'VIOLATED', 2: 'BREACHED', 3: 'APPOINTEDBASEDON'}
    NEGATIVE = 0
    VIOLATED = 1
    BREACHED = 2
    APPOINTEDBASEDON = 3

    # Labeling Functions
    # Capture "violated" relation between ORG and ARTICLE
    violated = {"violated"}
    @labeling_function(resources=dict(violated=violated), pre=[get_text_between])
    def lf_violated(x, violated):
        return VIOLATED if len(violated.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "breached" relation between ORG and ARTICLE
    breached = {"breached", "breach"}
    @labeling_function(resources=dict(breached=breached), pre=[get_text_between])
    def lf_breached(x, breached):
        return BREACHED if len(breached.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "appointed based on" relation between ORG and ARTICLE
    appointed = {"appointed"}
    @labeling_function(resources=dict(appointed=appointed), pre=[get_text_between])
    def lf_appointed(x, appointed):
        return APPOINTEDBASEDON if len(appointed.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_violated, lf_breached, lf_appointed
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

def get_ORG_RULE(df):
    # Example = "Crompton Grp. violated Rule 7"

    df_train = create_tag_df(df, "ORG", "RULE")

    # Set Classes for relations (e.g. violated etc.)
    labels = {0: 'NEGATIVE', 1: 'VIOLATED', 2: 'BREACHED', 3: 'APPOINTEDBASEDON'}
    NEGATIVE = 0
    VIOLATED = 1
    BREACHED = 2
    APPOINTEDBASEDON = 3

    # Labeling Functions
    # Capture "violated" relation between ORG and RULE
    violated = {"violated", "breached", "in breach"}
    @labeling_function(resources=dict(violated=violated), pre=[get_text_between])
    def lf_violated(x, violated):
        return VIOLATED if len(violated.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "breached" relation between ORG and RULE
    breached = {"breached", "breach"}
    @labeling_function(resources=dict(breached=breached), pre=[get_text_between])
    def lf_breached(x, breached):
        return BREACHED if len(breached.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "appointed based on" relation between ORG and RULE
    appointed = {"appointed"}
    @labeling_function(resources=dict(appointed=appointed), pre=[get_text_between])
    def lf_appointed(x, appointed):
        return APPOINTEDBASEDON if len(appointed.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_violated, lf_breached, lf_appointed
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

def get_COURT_PERSON(df):
    # Example = "The European Court of Human Rights (Second Section), sitting as a Chamber composed of: Jon Fridrik Kjølbro, President,	Carlo Ranzoni"

    df_train = create_tag_df(df, ["COURT"],
                             ['JUDGE', 'REGISTRAR', 'SECRETARY', 'LAWYER', 'DEFENDANT', 'INVESTIGATORS', 'PROSECUTOR', 'PERSON'])

    # Set Classes for relations (e.g. chambercomposition, convicted etc.)
    labels = {0: 'NEGATIVE', 1: 'CHAMBERCOMPOSITION', 2: 'CONVICTED', 3: 'INDICTMENTAGAINST'}
    NEGATIVE = 0
    CHAMBERCOMPOSITION = 1
    CONVICTED = 2
    INDICTMENTAGAINST = 3

    # Labeling Functions
    # Capture "Chamber composition" relation between Court and Person
    composed = {"composed", "constituted", "comprised", "sitting as"}
    @labeling_function(resources=dict(composed=composed), pre=[get_text_between])
    def lf_composed(x, composed):
        return CHAMBERCOMPOSITION if len(composed.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "convicted of" relation between Court and Person
    convicted = {"convicted"}
    @labeling_function(resources=dict(convicted=convicted), pre=[get_text_between])
    def lf_convicted(x, convicted):
        return CONVICTED if len(convicted.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "indictment against" relation between Court and Person
    indictment = {"indictment against"}
    @labeling_function(resources=dict(indictment=indictment), pre=[get_text_between])
    def lf_indictment(x, indictment):
        return INDICTMENTAGAINST if len(indictment.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_composed, lf_convicted, lf_indictment
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

def get_COURT_ARTICLE(df):
    # Example = "Conversely, in Bazo González v. Spain (no. 30643/04, 16 December 2008), the Court found that there had not been a violation of Article 6"

    df_train = create_tag_df(df, "COURT", "ARTICLE")

    # Set Classes for relations (e.g. foundviolation, foundnoviolation etc.)
    labels = {0: 'NEGATIVE', 1: 'FOUNDVIOLATION', 2: 'FOUNDNOVIOLATION', 3: 'CITED'}
    NEGATIVE = 0
    FOUNDVIOLATION = 1
    FOUNDNOVIOLATION = 2
    CITED = 3

    # Labeling Functions
    # Capture "violation" relation between COURT and ARTICLE
    violation = {"been a violation", "violation","violated"}
    @labeling_function(resources=dict(violation=violation), pre=[get_text_between])
    def lf_violation(x, violation):
        return FOUNDVIOLATION if len(violation.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "no violation" relation between COURT and ARTICLE
    noviolation = {"not been a violation", "been no violation", "no violation of"}
    @labeling_function(resources=dict(noviolation=noviolation), pre=[get_text_between])
    def lf_noviolation(x, noviolation):
        return FOUNDNOVIOLATION if len(noviolation.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "cited" relation between COURT and ARTICLE
    cited = {"cited"}
    @labeling_function(resources=dict(cited=cited), pre=[get_text_between])
    def lf_cited(x, cited):
        return CITED if len(cited.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_violation, lf_noviolation, lf_cited
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

def get_COURT_PROTOCOL(df):
    # Example = "The Court holds that there has been a violation of Article 1 of Protocol No. 1 to the Convention"

    df_train = create_tag_df(df, "COURT", "PROTOCOL")

    # Set Classes for relations (e.g. foundviolation, foundnoviolation etc.)
    labels = {0: 'NEGATIVE', 1: 'FOUNDVIOLATION', 2: 'FOUNDNOVIOLATION', 3: 'CITED', 4: 'HOLDSVIOLATION'}
    NEGATIVE = 0
    FOUNDVIOLATION = 1
    FOUNDNOVIOLATION = 2
    CITED = 3
    HOLDSVIOLATION = 4

    # Labeling Functions
    # Capture "found violation" relation between COURT and PROTOCOL
    violation = {"been a violation"}
    @labeling_function(resources=dict(violation=violation), pre=[get_text_between])
    def lf_violation(x, violation):
        return FOUNDVIOLATION if len(violation.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "found no violation" relation between COURT and PROTOCOL
    noviolation = {"not been a violation", "been no violation", "no violation of"}
    @labeling_function(resources=dict(noviolation=noviolation), pre=[get_text_between])
    def lf_noviolation(x, noviolation):
        return FOUNDNOVIOLATION if len(noviolation.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "cited" relation between COURT and PROTOCOL
    cited = {"cited"}
    @labeling_function(resources=dict(cited=cited), pre=[get_text_between])
    def lf_cited(x, cited):
        return CITED if len(cited.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "holds violation" relation between COURT and PROTOCOL
    violation = {"holds violation"}
    @labeling_function(resources=dict(violation=violation), pre=[get_text_between])
    def lf_holdviolation(x, violation):
        return HOLDSVIOLATION if len(violation.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_violation, lf_noviolation, lf_cited, lf_holdviolation
    ]

    # Apply them
    applier = PandasLFApplier(lfs)
    L_train = applier.apply(df_train)

    # Train labelingmodel
    label_model = LabelModel(cardinality=5, verbose=True)
    label_model.fit(L_train, n_epochs=5000, log_freq=500, seed=12345)

    probs_train = label_model.predict_proba(L_train)
    preds_train = probs_to_preds(probs_train)

    return list(map(lambda x: labels[x], preds_train))

def get_COURT_RULE(df):
    df_train = create_tag_df(df, "COURT", "RULE")

    # Set Classes for relations (e.g. foundviolation, foundnoviolation etc.)
    labels = {0: 'NEGATIVE', 1: 'FOUNDVIOLATION', 2: 'FOUNDNOVIOLATION', 3: 'CITED'}
    NEGATIVE = 0
    FOUNDVIOLATION = 1
    FOUNDNOVIOLATION = 2
    CITED = 3

    # Labeling Functions
    # Capture "violation" relation between COURT and RULE
    violation = {"been a violation"}
    @labeling_function(resources=dict(violation=violation), pre=[get_text_between])
    def lf_violation(x, violation):
        return FOUNDVIOLATION if len(violation.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "no violation" relation between COURT and RULE
    noviolation = {"not been a violation", "been no violation", "no violation of"}
    @labeling_function(resources=dict(noviolation=noviolation), pre=[get_text_between])
    def lf_noviolation(x, noviolation):
        return FOUNDNOVIOLATION if len(noviolation.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "cited" relation between COURT and RULE
    cited = {"cited"}
    @labeling_function(resources=dict(cited=cited), pre=[get_text_between])
    def lf_cited(x, cited):
        return CITED if len(cited.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_violation, lf_noviolation, lf_cited
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

def get_COURT_DATE(df):
    # Example = "The Court acceded to the request on 24 November"

    df_train = create_tag_df(df, "COURT", "DATE")

    # Set Classes for relations (e.g. accededon etc.)
    labels = {0: 'NEGATIVE', 1: 'ACCEDEDREQUEST', 2:'DISMISSEDON', 3:'RATIFIEDON'}
    NEGATIVE = 0
    ACCEDEDREQUEST = 1
    DISMISSEDON = 2
    RATIFIEDON = 3

    # Labeling Functions
    # Capture "acceded request on" relation between COURT and DATE
    acceded = {"acceded request"}
    @labeling_function(resources=dict(acceded=acceded), pre=[get_text_between])
    def lf_acceded(x, acceded):
        return ACCEDEDREQUEST if len(acceded.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "dismissed on" relation between COURT and DATE
    dismissedon = {"dismissed"}
    @labeling_function(resources=dict(dismissedon=dismissedon), pre=[get_text_between])
    def lf_dismissedon(x, dismissedon):
        return DISMISSEDON if len(dismissedon.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "ratified on" relation between COURT and DATE
    ratify = {"ratified"}
    @labeling_function(resources=dict(ratify=ratify), pre=[get_text_between])
    def lf_ratify(x, ratify):
        return RATIFIEDON if len(ratify.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_acceded, lf_dismissedon, lf_ratify
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

def get_NORP_RULE(df):
    # Example = "the Chamber decided under Rule 50 to relinquish jurisdiction forthwith in favour of the plenary Court"

    df_train = create_tag_df(df, "NORP", "RULE")

    # Set Classes for relations (e.g. decisionunder etc.)
    labels = {0: 'NEGATIVE', 1: 'DECISIONUNDER', 2: 'VIOLATED', 3: 'APPOINTEDBASEDON'}
    NEGATIVE = 0
    DECISIONUNDER = 1
    VIOLATED = 2
    APPOINTEDBASEDON = 3

    # Labeling Functions
    # Capture "decision under" relation between NORP and the Rule for decision
    decisionunder = {"decided under"}
    @labeling_function(resources=dict(decisionunder=decisionunder), pre=[get_text_between])
    def lf_decisionunder(x, decisionunder):
        return DECISIONUNDER if len(decisionunder.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Labeling Functions
    # Capture "violated" relation between NORP and RULE
    violated = {"violated", "breached", "in breach"}
    @labeling_function(resources=dict(violated=violated), pre=[get_text_between])
    def lf_violated(x, violated):
        return VIOLATED if len(violated.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "appointed based on" relation between NORP and RULE
    appointed = {"appointed"}
    @labeling_function(resources=dict(appointed=appointed), pre=[get_text_between])
    def lf_appointed(x, appointed):
        return APPOINTEDBASEDON if len(appointed.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_decisionunder, lf_violated, lf_appointed
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

def get_NORP_ARTICLE(df):
    # Example = "the Chamber decided under Article 51 to relinquish jurisdiction forthwith in favour of the plenary Court"

    df_train = create_tag_df(df, "NORP", "ARTICLE")

    # Set Classes for relations (e.g. decisionunder etc.)
    labels = {0: 'NEGATIVE', 1: 'DECISIONUNDER', 2: 'VIOLATED', 3: 'APPOINTEDBASEDON'}
    NEGATIVE = 0
    DECISIONUNDER = 1
    VIOLATED = 2
    APPOINTEDBASEDON = 3

    # Labeling Functions
    # Capture "decision under" relation between NORP and ARTICLE
    decisionunder = {"decided under"}
    @labeling_function(resources=dict(decisionunder=decisionunder), pre=[get_text_between])
    def lf_decisionunder(x, decisionunder):
        return DECISIONUNDER if len(decisionunder.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Labeling Functions
    # Capture "violated" relation between NORP and ARTICLE
    violated = {"violated", "breached", "in breach"}
    @labeling_function(resources=dict(violated=violated), pre=[get_text_between])
    def lf_violated(x, violated):
        return VIOLATED if len(violated.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "appointed based on" relation between NORP and ARTICLE
    appointed = {"appointed"}
    @labeling_function(resources=dict(appointed=appointed), pre=[get_text_between])
    def lf_appointed(x, appointed):
        return APPOINTEDBASEDON if len(appointed.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_decisionunder, lf_violated, lf_appointed
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

def get_NORP_PERSON(df):
    # Example = "The European Court of Human Rights (Second Section), sitting as a Chamber composed of: Jon Fridrik Kjølbro, President,	Carlo Ranzoni"

    df_train = create_tag_df(df, ["NORP"],
                             ['JUDGE', 'REGISTRAR', 'SECRETARY', 'LAWYER', 'DEFENDANT', 'INVESTIGATORS', 'PROSECUTOR', 'PERSON'])

    # Set Classes for relations (e.g. chambercomposition, appointed etc.)
    labels = {0: 'NEGATIVE', 1: 'CHAMBERCOMPOSITION', 2: 'REPRESENTEDBY', 3: 'APPOINTED'}
    NEGATIVE = 0
    CHAMBERCOMPOSITION = 1
    REPRESENTEDBY = 2
    APPOINTED = 3

    # Labeling Functions
    # Capture "Chamber composition" relation between NORP and Person
    composed = {"composed", "constituted", "comprised", "sitting as a"}
    @labeling_function(resources=dict(composed=composed), pre=[get_text_between])
    def lf_composed(x, composed):
        return CHAMBERCOMPOSITION if len(composed.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "represented by" relation between NORP and Person
    represented = {"represented by"}
    @labeling_function(resources=dict(represented=represented), pre=[get_text_between])
    def lf_represented(x, represented):
        return REPRESENTEDBY if len(represented.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "appointed" relation between NORP and Person
    appointed = {"appointed"}
    @labeling_function(resources=dict(appointed=appointed), pre=[get_text_between])
    def lf_appointed(x, appointed):
        return APPOINTED if len(appointed.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_composed, lf_represented, lf_appointed
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

def get_APPLICATION_GPE(df):
    # Example = "The case originated in two applications (nos. 7299/75 and 7496/76) against Belgium"

    df_train = create_tag_df(df, "APPLICATION", "GPE")

    # Set Classes for relations (e.g. against etc.)
    labels = {0: 'NEGATIVE', 1: 'AGAINST', 2:'FILEDIN', 3:'DISMISSEDIN'}
    NEGATIVE = 0
    AGAINST = 1
    FILEDIN = 2
    DISMISSEDIN = 3

    # Labeling Functions
    # Capture "against" relation between Application and GPE including countries, states etc
    against = {"against"}
    @labeling_function(resources=dict(against=against), pre=[get_text_between])
    def lf_against(x, against):
        return AGAINST if len(against.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "filed in" relation between Application and GPE including countries, states etc
    filed = {"filed in", "filed"}
    @labeling_function(resources=dict(filed=filed), pre=[get_text_between])
    def lf_filed(x, filed):
        return FILEDIN if len(filed.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "dismissed in" relation between Application and GPE including countries, states etc
    dismissed = {"dismissed in"}
    @labeling_function(resources=dict(dismissed=dismissed), pre=[get_text_between])
    def lf_dismissed(x, dismissed):
        return DISMISSEDIN if len(dismissed.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_against, lf_filed, lf_dismissed
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

def get_APPLICATION_PERSON(df):
    # Example = "The case originated in an application (no. 10929/84) against Denmark lodged with the Commission in 1984 by Mr Jon Nielsen"

    df_train = create_tag_df(df, ["APPLICATION"],
                             ['JUDGE', 'REGISTRAR', 'SECRETARY', 'LAWYER', 'DEFENDANT', 'INVESTIGATORS', 'PROSECUTOR', 'PERSON'])

    # Set Classes for relations (e.g. lodgedby etc.)
    labels = {0: 'NEGATIVE', 1: 'LODGEDBY', 2:'LODGEDAGAINST', 3:'CONCERNING'}
    NEGATIVE = 0
    LODGEDBY = 1
    LODGEDAGAINST = 2
    CONCERNING = 3

    # Labeling Functions
    # Capture "lodged by" relation between Application and Person
    lodgedby = {"by", "lodged by"}
    @labeling_function(resources=dict(lodgedby=lodgedby), pre=[get_text_between])
    def lf_lodgedby(x, lodgedby):
        return LODGEDBY if len(lodgedby.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "lodged against" relation between Application and Person
    lodgedagainst = {"against", "lodged against"}
    @labeling_function(resources=dict(lodgedagainst=lodgedagainst), pre=[get_text_between])
    def lf_lodgedagainst(x, lodgedagainst):
        return LODGEDAGAINST if len(lodgedagainst.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "concerning" relation between Application and Person
    concerning = {"concerning"}
    @labeling_function(resources=dict(concerning=concerning), pre=[get_text_between])
    def lf_concerning(x, concerning):
        return CONCERNING if len(concerning.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_lodgedby, lf_lodgedagainst, lf_concerning
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

def get_APPLICATION_DATE(df):
    # Example = "The case originated in an application (no. 10929/84) against Denmark lodged with the Commission in 1984 by Mr Jon Nielsen"

    df_train = create_tag_df(df, "APPLICATION", "DATE")

    # Set Classes for relations (e.g. lodgedin, lodgedon etc.)
    labels = {0: 'NEGATIVE', 1: 'LODGEDIN', 2:'SUBMITTEDON', 3:'RELIEDON'}
    NEGATIVE = 0
    LODGEDIN = 1
    SUBMITTEDON = 2
    RELIEDON = 3

    # Labeling Functions
    # Capture "lodged in/on" relation between Application and Date
    lodgedin = {"lodged"}
    @labeling_function(resources=dict(lodgedin=lodgedin), pre=[get_text_between])
    def lf_lodgedin(x, lodgedin):
        return LODGEDIN if len(lodgedin.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "submitted on" relation between Application and Date
    submittedon = {"submitted on"}
    @labeling_function(resources=dict(submittedon=submittedon), pre=[get_text_between])
    def lf_submittedon(x, submittedon):
        return SUBMITTEDON if len(submittedon.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "rely on" relation between Application and Date
    relyon = {"rely on", "relied on"}
    @labeling_function(resources=dict(relyon=relyon), pre=[get_text_between])
    def lf_relyon(x, relyon):
        return RELIEDON if len(relyon.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_lodgedin, lf_submittedon, lf_relyon
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

def get_APPLICATION_ARTICLE(df):
    # Example = "The case originated in an application (no. 10929/84) against Denmark lodged with the Commission by Mr Jon Nielsen under Article 25"

    df_train = create_tag_df(df, "APPLICATION", "ARTICLE")

    # Set Classes for relations (e.g. lodgedunder etc.)
    labels = {0: 'NEGATIVE', 1: 'LODGEDUNDER', 2:'INVOKED', 3:'REITERATED'}
    NEGATIVE = 0
    LODGEDUNDER = 1
    INVOKED = 2
    REITERATED = 3

    # Labeling Functions
    # Capture "lodged under" relation between Application and Article
    lodgedunder = {"lodged"}
    @labeling_function(resources=dict(lodgedunder=lodgedunder), pre=[get_text_between])
    def lf_lodgedunder(x, lodgedunder):
        return LODGEDUNDER if len(lodgedunder.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "invoke" relation between Application and Article
    invoked = {"invoked", "invoke"}
    @labeling_function(resources=dict(invoked=invoked), pre=[get_text_between])
    def lf_invoked(x, invoked):
        return INVOKED if len(invoked.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "reiterated" relation between Application and Article
    reiterated = {"reiterated", "reiterate"}
    @labeling_function(resources=dict(reiterated=reiterated), pre=[get_text_between])
    def lf_reiterated(x, reiterated):
        return REITERATED if len(reiterated.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_lodgedunder, lf_invoked, lf_reiterated
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

def get_CASE_ORG(df):
    # Example = "The Airey case was referred to the Court by the European Commission of Human Rights"

    df_train = create_tag_df(df, "CASE", "ORG")

    # Set Classes for relations (e.g. referredby etc.)
    labels = {0: 'NEGATIVE', 1: 'REFERREDBY', 2: 'REFERREDTO', 3:'FILEDAGAINST'}
    NEGATIVE = 0
    REFERREDBY = 1
    REFERREDTO = 2
    FILEDAGAINST = 3

    # Labeling Functions
    # Capture "referred by" relation between Case and Organization
    referredby = {"referred by", "referred to the Court by"}
    @labeling_function(resources=dict(referredby=referredby), pre=[get_text_between])
    def lf_referredby(x, referredby):
        return REFERREDBY if len(referredby.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "referred to" relation between Case and Organization
    referredto = {"referred to"}
    @labeling_function(resources=dict(referredto=referredto), pre=[get_text_between])
    def lf_referredto(x, referredto):
        return REFERREDTO if len(referredto.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "referred to" relation between Case and Organization
    filedagainst = {"filed against"}
    @labeling_function(resources=dict(filedagainst=filedagainst), pre=[get_text_between])
    def lf_filedagainst(x, filedagainst):
        return FILEDAGAINST if len(filedagainst.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_referredby, lf_referredto, lf_filedagainst
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

def get_CASE_DATE(df):
    # Example = "The present case was referred to the Court on 12 March 1982 by the European Commission of Human Rights ("the Commission")."

    df_train = create_tag_df(df, "CASE", "DATE")

    # Set Classes for relations (e.g. referredon etc.)
    labels = {0: 'NEGATIVE', 1: 'REFERREDON', 2: 'ORIGINATEDON', 3:'SETTLEDON'}
    NEGATIVE = 0
    REFERREDON = 1
    ORIGINATEDON = 2
    SETTLEDON = 3

    # Labeling Functions
    # Capture "referred on" relation between Case and DATE
    referredon = {"referred"}
    @labeling_function(resources=dict(referredon=referredon), pre=[get_text_between])
    def lf_referredon(x, referredon):
        return REFERREDON if len(referredon.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "originated on" relation between Case and DATE
    originatedon = {"originated"}
    @labeling_function(resources=dict(originatedon=originatedon), pre=[get_text_between])
    def lf_originatedon(x, originatedon):
        return ORIGINATEDON if len(originatedon.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "originated on" relation between Case and DATE
    settledon = {"settled on"}
    @labeling_function(resources=dict(settledon=settledon), pre=[get_text_between])
    def lf_settledon(x, settledon):
        return SETTLEDON if len(settledon.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_referredon, lf_originatedon, lf_settledon
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

def get_CASE_COURT(df):
    # Example = "The case was accordingly remitted to the Erzurum Assize Court."

    df_train = create_tag_df(df, "CASE", "COURT")

    # Set Classes for relations (e.g. referredto etc.)
    labels = {0: 'NEGATIVE', 1: 'REFERREDTO', 2: 'REMITTEDTO', 3:'SUBMITTEDTO'}
    NEGATIVE = 0
    REFERREDTO = 1
    REMITTEDTO = 2
    SUBMITTEDTO = 3

    # Labeling Functions
    # Capture "referred to" relation between Case and COURT
    referredto = {"referred to"}
    @labeling_function(resources=dict(referredto=referredto), pre=[get_text_between])
    def lf_referredto(x, referredto):
        return REFERREDTO if len(referredto.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "remitted to" relation between Case and COURT
    remitted = {"remitted"}
    @labeling_function(resources=dict(remitted=remitted), pre=[get_text_between])
    def lf_remitted(x, remitted):
        return REMITTEDTO if len(remitted.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "referred by" relation between Case and COURT
    submittedto = {"submitted to"}
    @labeling_function(resources=dict(submittedto=submittedto), pre=[get_text_between])
    def lf_submittedto(x, submittedto):
        return SUBMITTEDTO if len(submittedto.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_referredto, lf_remitted, lf_submittedto
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

def get_CASE_PERSON(df):
    # Example = "The case originated in an application against this State lodged with the Commission by an Austrian citizen, Mr. Gustav Adolf"

    df_train = create_tag_df(df, ["CASE"],
                             ['JUDGE', 'REGISTRAR', 'SECRETARY', 'LAWYER', 'DEFENDANT', 'INVESTIGATORS', 'PROSECUTOR', 'PERSON'])

    # Set Classes for relations (e.g. lodgedby, sentenced, etc.)
    labels = {0: 'NEGATIVE', 1: 'LODGEDBY', 2:'LODGEDAGAINST', 3:'FILEDAGAINST'}
    NEGATIVE = 0
    LODGEDBY = 1
    LODGEDAGAINST = 2
    FILEDAGAINST = 3

    # Labeling Functions
    # Capture "lodged by" relation between Case and Person
    lodgedby = {"lodged by"}
    @labeling_function(resources=dict(lodgedby=lodgedby), pre=[get_text_between])
    def lf_lodgedby(x, lodgedby):
        return LODGEDBY if len(lodgedby.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "lodged against" relation between Case and Person
    lodgedagainst = {"lodged against"}
    @labeling_function(resources=dict(lodgedagainst=lodgedagainst), pre=[get_text_between])
    def lf_lodgedagainst(x, lodgedagainst):
        return LODGEDAGAINST if len(lodgedagainst.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "referred to" relation between Case and Person
    filedagainst = {"filed against"}
    @labeling_function(resources=dict(filedagainst=filedagainst), pre=[get_text_between])
    def lf_filedagainst(x, filedagainst):
        return FILEDAGAINST if len(filedagainst.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_lodgedby, lf_lodgedagainst, lf_filedagainst
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

def get_CASE_APPLICATION(df):
    # Example = "The case originated in an application (no. 10929/84) against Denmark lodged with the Commission in 1984 by Mr Jon Nielsen"

    df_train = create_tag_df(df, "CASE", "APPLICATION")

    # Set Classes for relations (e.g. originatedin, sentenced, etc.)
    labels = {0: 'NEGATIVE', 1: 'ORIGINATEDIN', 2:'REFERREDTO', 3: 'INCLUDED'}
    NEGATIVE = 0
    ORIGINATEDIN = 1
    REFERREDTO = 2
    INCLUDED = 3

    # Labeling Functions
    # Capture "originated in" relation between Case and Application
    originatedin = {"originated in", "originated", "origin"}
    @labeling_function(resources=dict(originatedin=originatedin), pre=[get_text_between])
    def lf_orginatedin(x, originatedin):
        return ORIGINATEDIN if len(originatedin.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "referred to" relation between Case and Application
    referredto = {"referred to"}
    @labeling_function(resources=dict(referredto=referredto), pre=[get_text_between])
    def lf_referredto(x, referredto):
        return REFERREDTO if len(referredto.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "referred to" relation between Case and Application
    included = {"included"}
    @labeling_function(resources=dict(included=included), pre=[get_text_between])
    def lf_included(x, included):
        return INCLUDED if len(included.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_orginatedin, lf_referredto, lf_included
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

def get_PROSECUTOR_PERSON(df):
    # Example = "The public prosecutor at the Istanbul State Security Court filed a bill of indictment against the applicant"

    df_train = create_tag_df(df, "PROSECUTOR", "PERSON")

    # Set Classes for relations (e.g. indictment against etc.)
    labels = {0: 'NEGATIVE', 1: 'INDICTMENTAGAINST', 2:'AUTHORISED', 3:'PROSECUTED'}
    NEGATIVE = 0
    INDICTMENTAGAINST = 1
    AUTHORISED = 2
    PROSECUTED = 3

    # Labeling Functions
    # Capture "indictment against" relation between PROSECUTOR and Person
    indictment = {"indictment against"}
    @labeling_function(resources=dict(indictment=indictment), pre=[get_text_between])
    def lf_indictment(x, indictment):
        return INDICTMENTAGAINST if len(indictment.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "authorized" relation between PROSECUTOR and PERSON
    authorised = {"authorized", "authorised"}
    @labeling_function(resources=dict(authorised=authorised), pre=[get_text_between])
    def lf_authorised(x, authorised):
        return AUTHORISED if len(authorised.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "submitted" relation between PROSECUTOR and PERSON
    prosecuted = {"prosecuted"}
    @labeling_function(resources=dict(prosecuted=prosecuted), pre=[get_text_between])
    def lf_prosecuted(x, prosecuted):
        return PROSECUTED if len(prosecuted.intersection(set(x.between_tokens))) > 0 else NEGATIVE

        # Use all labeling functions

    lfs = [
        lf_indictment, lf_authorised, lf_prosecuted
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

def get_PROSECUTOR_ORG(df):
    df_train = create_tag_df(df, "PROSECUTOR", "ORG")

    # Set Classes for relations (e.g. indictment against etc.)
    labels = {0: 'NEGATIVE', 1: 'INDICTMENTAGAINST', 2:'AUTHORISED', 3:'PROSECUTED'}
    NEGATIVE = 0
    INDICTMENTAGAINST = 1
    AUTHORISED = 2
    PROSECUTED = 3

    # Labeling Functions
    # Capture "indictment against" relation between PROSECUTOR and ORG
    indictment = {"indictment against"}
    @labeling_function(resources=dict(indictment=indictment), pre=[get_text_between])
    def lf_indictment(x, indictment):
        return INDICTMENTAGAINST if len(indictment.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "authorized" relation between PROSECUTOR and ORG
    authorised = {"authorized", "authorised"}
    @labeling_function(resources=dict(authorised=authorised), pre=[get_text_between])
    def lf_authorised(x, authorised):
        return AUTHORISED if len(authorised.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "submitted" relation between PROSECUTOR and ORG
    prosecuted = {"prosecuted"}
    @labeling_function(resources=dict(prosecuted=prosecuted), pre=[get_text_between])
    def lf_prosecuted(x, prosecuted):
        return PROSECUTED if len(prosecuted.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_indictment, lf_authorised, lf_prosecuted
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

def get_PROSECUTOR_NORP(df):
    df_train = create_tag_df(df, "PROSECUTOR", "NORP")

    # Set Classes for relations (e.g. INDICTMENTAGAINST etc.)
    labels = {0: 'NEGATIVE', 1: 'INDICTMENTAGAINST', 2:'AUTHORISED', 3:'ORDEREDAGAINST'}
    NEGATIVE = 0
    INDICTMENTAGAINST = 1
    AUTHORISED = 2
    ORDEREDAGAINST = 3

    # Labeling Functions
    # Capture "indictment against" relation between PROSECUTOR and NORP
    indictment = {"indictment against"}
    @labeling_function(resources=dict(indictment=indictment), pre=[get_text_between])
    def lf_indictment(x, indictment):
        return INDICTMENTAGAINST if len(indictment.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "authorized" relation between PROSECUTOR and NORP
    authorised = {"authorized", "authorised"}
    @labeling_function(resources=dict(authorised=authorised), pre=[get_text_between])
    def lf_authorised(x, authorised):
        return AUTHORISED if len(authorised.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "ordered against" relation between PROSECUTOR and NORP
    orderedagainst = {"ordered against"}
    @labeling_function(resources=dict(orderedagainst=orderedagainst), pre=[get_text_between])
    def lf_orderedagainst(x, orderedagainst):
        return ORDEREDAGAINST if len(orderedagainst.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_indictment, lf_authorised, lf_orderedagainst
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

def get_PROSECUTOR_COURT(df):
    df_train = create_tag_df(df, "PROSECUTOR", "COURT")

    # Set Classes for relations (e.g. REQUESTED etc.)
    labels = {0: 'NEGATIVE', 1: 'REQUESTED', 2:'NOTED', 3:'SUBMITTED'}
    NEGATIVE = 0
    REQUESTED = 1
    NOTED = 2
    SUBMITTED = 3

    # Labeling Functions
    # Capture "requested" relation between PROSECUTOR and COURT
    requested = {"requested", "petitioned"}
    @labeling_function(resources=dict(requested=requested), pre=[get_text_between])
    def lf_requested(x, requested):
        return REQUESTED if len(requested.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "noted" relation between PROSECUTOR and COURT
    noted = {"noted"}
    @labeling_function(resources=dict(noted=noted), pre=[get_text_between])
    def lf_noted(x, noted):
        return NOTED if len(noted.intersection(set(x.between_tokens))) > 0 else NEGATIVE


    # Capture "submitted" relation between PROSECUTOR and COURT
    submitted = {"submitted"}
    @labeling_function(resources=dict(submitted=submitted), pre=[get_text_between])
    def lf_submitted(x, submitted):
        return SUBMITTED if len(submitted.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_requested, lf_noted, lf_submitted
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

def get_PROSECUTOR_DATE(df):
    df_train = create_tag_df(df, "PROSECUTOR", "DATE")

    # Set Classes for relations (e.g. REQUESTEDON etc.)
    labels = {0: 'NEGATIVE', 1: 'REQUESTEDON', 2:'SUBMITTED', 3:'PROSECUTEDON'}
    NEGATIVE = 0
    REQUESTEDON = 1
    SUBMITTED = 2
    PROSECUTEDON = 3

    # Labeling Functions
    # Capture "requested on" relation between PROSECUTOR and DATE
    requested = {"requested", "petitioned"}
    @labeling_function(resources=dict(requested=requested), pre=[get_text_between])
    def lf_requestedon(x, requested):
        return REQUESTEDON if len(requested.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "submitted" relation between PROSECUTOR and DATE
    submitted = {"submitted"}
    @labeling_function(resources=dict(submitted=submitted), pre=[get_text_between])
    def lf_submitted(x, submitted):
        return SUBMITTED if len(submitted.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Capture "submitted" relation between PROSECUTOR and DATE
    prosecuted = {"prosecuted"}
    @labeling_function(resources=dict(prosecuted=prosecuted), pre=[get_text_between])
    def lf_prosecuted(x, prosecuted):
        return PROSECUTEDON if len(prosecuted.intersection(set(x.between_tokens))) > 0 else NEGATIVE

    # Use all labeling functions
    lfs = [
        lf_requestedon, lf_submitted, lf_prosecuted
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

