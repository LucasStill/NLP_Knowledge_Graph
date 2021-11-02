import requests
from tqdm import tqdm
from pathlib import Path
import os

def get_ids():
    # Basic query_url
    query_url = "https://hudoc.echr.coe.int/app/query/results?query=contentsitename%3AECHR%20AND%20(NOT%20(doctype%3DPR%20OR%20doctype%3DHFCOMOLD%20OR%20doctype%3DHECOMOLD))%20AND%20((languageisocode%3D%22ENG%22))%20AND%20((documentcollectionid%3D%22GRANDCHAMBER%22)%20OR%20(documentcollectionid%3D%22CHAMBER%22))&select=sharepointid,Rank,ECHRRanking,languagenumber,itemid,docname,doctype,application,appno,conclusion,importance,originatingbody,typedescription,kpdate,kpdateAsText,documentcollectionid,documentcollectionid2,languageisocode,extractedappno,isplaceholder,doctypebranch,respondent,ecli,appnoparts,sclappnos,echradvopidentifier,echradvopstatus&sort=&rankingModelId=11111111-0000-0000-0000-000000000000"

    # Query no entries (to see how many exist entries exist)
    req = requests.get("https://hudoc.echr.coe.int/app/query/results?query=contentsitename%3AECHR%20AND%20(NOT%20(doctype%3DPR%20OR%20doctype%3DHFCOMOLD%20OR%20doctype%3DHECOMOLD))%20AND%20((languageisocode%3D%22ENG%22))%20AND%20((documentcollectionid%3D%22GRANDCHAMBER%22)%20OR%20(documentcollectionid%3D%22CHAMBER%22))&select=sharepointid,Rank,ECHRRanking,languagenumber,itemid,docname,doctype,application,appno,conclusion,importance,originatingbody,typedescription,kpdate,kpdateAsText,documentcollectionid,documentcollectionid2,languageisocode,extractedappno,isplaceholder,doctypebranch,respondent,ecli,appnoparts,sclappnos,echradvopidentifier,echradvopstatus&sort=&rankingModelId=11111111-0000-0000-0000-000000000000")
    number_of_entries = req.json()['resultcount']

    # For every entries, load them and add ids to list
    ids_lst = []
    for i in tqdm(range(0, number_of_entries, 500), desc="Loading all ids"):
        # Get request and transform to Jason
        req_json = requests.get(query_url + "&start={}&length=500".format(i)).json()

        # Add id to list
        for entry in req_json['results']:
            ids_lst.append(entry['columns']['itemid'])

    return ids_lst

def get_docx(ids_lst):
    # Query link to get docx files
    docx_url = "http://hudoc.echr.coe.int/app/conversion/docx/?library=ECHR&filename=thank_you.docx&id="

    # Create a folder "Files" if it doesn't exist already
    Path("/Files").mkdir(parents=True, exist_ok=True)

    # For each id
    for entry_id in tqdm(ids_lst, "Downloading all files"):
        # Create a filename
        fn = os.path.join('Files', "{}.docx".format(entry_id))

        # If the file does not already exist
        if not os.path.isfile(fn):
            # Load request
            req = requests.get(docx_url + entry_id, stream = True)

            # Save request
            with open(fn, 'wb') as f:
                for chunk in req.iter_content(1024):
                    f.write(chunk)

def scrape_all():
    # Get all documents
    get_docx(get_ids())

scrape_all()
