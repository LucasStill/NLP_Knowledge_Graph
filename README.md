# Extracting SVOs from Legal Documents
## Introduction
In this project, Subject Verb Object triples (SVOs) will be extracted from legal documents. Deep-learning approaches have been used to extract these. To extract SVOs, the project has been split into two parts, Named Entitiy Recognition (NER) and Relation Extraction (RE). First Bi-LSTM models have been made for both NER and RE, as benchmark models. Finally, a Two Headed Bert Model (THBM) has been designed, to predict both NER and RE in a single model.

## Data
For this project the HUDOC dataset has been used. Since this dataset only contains raw data, we had to label the dataset as well in this project. First of all, the data had to be scraped. This code can be found in ```Scraper.py```. 

After the code has been scraped, the code need to be annotated for NER. To annotate this use ```MRP1_Hudoc_NER_annotation.ipynb``` for the regular expressions and merge these with the annotations from the pretrained model in ```xxx```. 

Once the NE are labelled correctly, use ```label_relations.py``` to label the relations accordingly. The labels / relations used and designed (for SNORKEL) are determined using ```ClausIE_notebook.ipynb```, which is a dependency parser, thus is able to extract all relations.

## Model
The dataset created in ```label_relations.py``` can be used in ```NER_LSTM.ipynb``` to train the Bi-LSTM for NER and ```train_re_model.py``` to train the Bi-LSTM for RE. Finally, ```MRP1_Two_headed_BERT.ipynb``` could be used to train the THBM. 
