import spacy
from spacy.lang.en import English
import networkx as nx
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def getSentences(text):
    nlp = English()
    #nlp.add_pipe(nlp.create_pipe('sentencizer'))
    nlp.add_pipe('sentencizer')
    document = nlp(text)
    #print([type(sent.text) for sent in document.sents])
    return [sent.text.strip() for sent in document.sents]

def printToken(token):
    print(token.text, "->", token.dep_)

def appendChunk(original, chunk):
    return original + ' ' + chunk

def isRelationCandidate(token):
    deps = ["ROOT", "adj", "attr", "agent", "amod"]
    return any(subs in token.dep_ for subs in deps)

def isConstructionCandidate(token):
    deps = ["compound", "prep", "conj", "mod"]
    return any(subs in token.dep_ for subs in deps)

def processSubjectObjectPairs(tokens):
    subject = ''
    object = ''
    relation = ''
    subjectConstruction = ''
    objectConstruction = ''
    for token in tokens:
        #printToken(token)
        if "punct" in token.dep_:
            continue
        if isRelationCandidate(token):
            relation = appendChunk(relation, token.lemma_)
        if isConstructionCandidate(token):
            if subjectConstruction:
                subjectConstruction = appendChunk(subjectConstruction, token.text)
            if objectConstruction:
                objectConstruction = appendChunk(objectConstruction, token.text)
        if "subj" in token.dep_:
            subject = appendChunk(subject, token.text)
            subject = appendChunk(subjectConstruction, subject)
            subjectConstruction = ''
        if "obj" in token.dep_:
            object = appendChunk(object, token.text)
            object = appendChunk(objectConstruction, object)
            objectConstruction = ''

    print (subject.strip(), ",", relation.strip(), ",", object.strip())
    return (subject.strip(), relation.strip(), object.strip())

def processSentence(sentence):
    tokens = nlp_model(sentence)
    return processSubjectObjectPairs(tokens)

def printGraph(triples):
    G = nx.Graph()
    for triple in triples:
        G.add_node(triple[0])
        G.add_node(triple[1])
        G.add_node(triple[2])
        G.add_edge(triple[0], triple[1])
        G.add_edge(triple[1], triple[2])

    pos = nx.spring_layout(G)
    plt.figure()
    nx.draw(G, pos, edge_color='black', width=1, linewidths=1,
            node_size=500, node_color='cornflowerblue', alpha=0.9,
            labels={node: node for node in G.nodes()})
    plt.axis('off')
    plt.show()

if __name__ == "__main__":

    text = "London is the capital and largest city of England and the United Kingdom. Standing on the River " \
           "Thames in the south-east of England, at the head of its 50-mile (80 km) estuary leading to " \
           "the North Sea, London has been a major settlement for two millennia. " \
           "Londinium was founded by the Romans. The City of London, " \
           "London's ancient core − an area of just 1.12 square miles (2.9 km2) and colloquially known as " \
           "the Square Mile − retains boundaries that follow closely its medieval limits." \
           "The City of Westminster is also an Inner London borough holding city status. " \
           "Greater London is governed by the Mayor of London and the London Assembly." \
           "London is located in the southeast of England." \
           "Westminster is located in London." \
           "London is the biggest city in Britain. London has a population of 7,172,036."

    # text = "The Court reiterates that by virtue of the essential function the press fulfils in a democracy, Article 10 of the Convention affords journalists protection, subject to the proviso that they act in good faith in order to provide accurate and reliable information in accordance with the tenets of responsible journalism (see, among other authorities, Pentikäinen v. Finland [GC], no. 11882/10, § 90, ECHR 2015). In considering the “duties and responsibilities” of a journalist, the potential impact of the medium concerned is an important factor and it is commonly acknowledged that the audiovisual media have often a much more immediate and powerful effect than the print media. The audiovisual media have means of conveying through images meanings which the print media are not able to impart. At the same time, the methods of objective and balanced reporting may vary considerably, depending among other things on the media in question. It is not for this Court, nor for the national courts for that matter, to substitute their own views for those of the press as to what technique of reporting should be adopted by journalists. In this context the Court reiterates that Article 10 protects not only the substance of the ideas and information expressed, but also the form in which they are conveyed (see Jersild, cited above, §§ 31). The punishment of a journalist for assisting in the dissemination of statements made by another person in an interview would seriously hamper the contribution of the press to discussion of matters of public interest and should not be envisaged unless there are particularly strong reasons for doing so (ibid., § 35, and Thoma, cited above, § 62). A general requirement for journalists systematically and formally to distance themselves from the content of a quotation that might insult or provoke others or damage their reputation is not reconcilable with the press’s role of providing information on current events, opinions and ideas (see Thoma, cited above, § 64)."
    # text = "On 11 February 2014 the Broadcasting Council issued a new decision in which it again concluded that the applicant company had breached the Broadcasting and Retransmission Act and fined it EUR 500. It held that the applicant company’s freedom of expression was to be restricted on the grounds of the ban on promoting drug use provided for in section 19(1)e) of the Broadcasting and Retransmission Act, which pursued the legitimate aim of protecting public order. That ban reflected the public interest in not publishing information which amounted to a positive assessment of drug use. Given the objective (strict) liability nature of the administrative offence, what was decisive in the case at hand was not whether the applicant company had aimed to promote drug use, but whether the programme, in the light of its content and the manner of processing the information, had had a promotional character. In the Broadcasting Council’s opinion, such was the case since X.’s comments had disseminated the idea that marijuana had a positive influence; the journalist’s comments had downplayed and justified them as being common, which went beyond a simple statement of views and beyond reproducing information that had already been publicly available. In that way, the applicant company had significantly interfered with the legitimate interests in protecting public order, health and morals, while the lowest possible fine had restricted its freedom of expression to a very little extent, which had made the interference fully proportionate."

    sentences = getSentences(text)
    nlp_model = spacy.load('en_core_web_sm')

    triples = []
    print (text)
    for sentence in sentences:
        triples.append(processSentence(sentence))

    printGraph(triples)
