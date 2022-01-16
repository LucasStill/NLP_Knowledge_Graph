# Extracting SVOs from Legal Documents
## Introduction
In this project, Subject Verb Object triples (SVOs) will be extracted from legal documents. Deep-learning approaches have been used to extract these. To extract SVOs, the project has been splitted into two parts, Named Entitiy Recognition (NER) and Relation Extraction (RE). First Bi-LSTM models have been made for both NER and RE, as benchmark models. Finally, a Two Headed Bert Model (THBM) has been designed, to predict both NER and RE in a single model.

## Data
For this project the HUDOC dataset has been used. Since this dataset only contains raw data, we had to label the dataset in this project as well. First of all, the data had to be scraped. This code can be found in ```Scraper.py```.
