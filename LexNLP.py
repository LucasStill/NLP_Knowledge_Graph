from tika import parser
import glob, os
import pandas as pd
from bs4 import BeautifulSoup
import codecs
import re
import numpy as np
import lexnlp.extract.en
import lexnlp.extract.en.entities.nltk_maxent
import lexnlp.extract.en.entities.stanford_ner
import lexnlp.extract.en.entities.company_detector
import lexnlp.extract.en.entities.nltk_re
import lexnlp.extract.en.amounts
import lexnlp.nlp.en.tokens

def load_txt(path):
    with open(path, encoding="utf8") as f:
        lines = f.read().replace('\n', '')
    return lines

text = "There are ten cows in the 2 acre pasture."
print(list(lexnlp.nlp.en.tokens.get_nouns(text)))

text = "This is Deutsche Bank Securities Inc."
print(list(lexnlp.extract.en.entities.nltk_maxent.get_companies(text)))
#print(list(lexnlp.extract.en.entities.company_detector.CompanyDetector.get_companies(text=text)))

text = "There are ten cows in the 2 acre pasture."
print(list(lexnlp.extract.en.amounts.get_amounts(text)))

path = r'./FIRSTSECTIONCASE.txt'
text = load_txt(path)

print("nltk_maxent")
print(list(lexnlp.extract.en.entities.nltk_maxent.get_companies(text)))
print(list(lexnlp.extract.en.entities.nltk_maxent.get_persons(text)))
print(list(lexnlp.extract.en.entities.nltk_maxent.get_geopolitical(text)))
print(list(lexnlp.extract.en.entities.nltk_maxent.get_parties_as(text)))

# print("stanford_ner")
# print(list(lexnlp.extract.en.entities.stanford_ner.get_persons(text)))
# print(list(lexnlp.extract.en.entities.stanford_ner.get_locations(text)))
# print(list(lexnlp.extract.en.entities.stanford_ner.get_organizations(text)))
