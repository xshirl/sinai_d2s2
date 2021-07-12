import os
import pandas as pd
from pandas.compat import StringIO
import numpy as np
from numpy import loadtxt
import sys
import json
from pprint import pprint
import objectpath
import csv
import re
import matplotlib.pyplot as plt
import json
import io
import http.client
import requests
from pprint import pprint
import itertools
import scipy as sp
from scipy.spatial import distance
from sklearn.metrics.pairwise import pairwise_distances
from clustergrammer import Network
#from clustergrammer_widget import *
from flask import Flask, render_template, request, redirect, jsonify, send_from_directory, abort, Response, send_file
import urllib.request
import h5py
import h5sparse
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix, vstack
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
# os.chdir('./scripts')
from .gvm_scripts import *
from .vae_scripts import *
import keras
from keras import backend as K
import tensorflow


app = Flask(__name__,static_url_path='/d2s2/static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.static_folder = 'static'

ENDPOINT = "/d2s2"
# load in the EMR Data (filtered > 200 in R code [Drug_diagnosis_test_code.R])
EMR_data = pd.read_csv(urllib.request.urlopen('https://amp.pharm.mssm.edu/lincs-playground/index.php/s/BJjCbVZDo4y0P7e/download'))
EMR_data_df = pd.DataFrame(EMR_data)
# implement the search from ICD9-do_id from the manual conversion
icd9_to_doid = pd.read_csv(urllib.request.urlopen('https://amp.pharm.mssm.edu/lincs-playground/index.php/s/qYCgxsmu16daKya/download'))
icd9_to_doid = pd.DataFrame(icd9_to_doid) # convert it to a data fram to drop unecessary rows
#icd9_to_doid # sanity check
icd9_to_doid_final = icd9_to_doid.drop(icd9_to_doid.columns[[0, 6, 7, 8, 9, 10, 11, 12, 13, 14]], axis = 1)
#icd9_to_doid_final # sanity check

### CREEDS DISEASE CARD PRELOAD
#L1000_CREEDS_Similar_Drug_output.txt
with urllib.request.urlopen('http://amp.pharm.mssm.edu/lincs-playground/index.php/s/ro0KDMjUpFypEdJ/download') as fd:
    similar_L1000_CREEDS_drugs = fd.read().splitlines()
    
# CREEDS up and down genes for disease signatures
#disease signature json
resp = urllib.request.urlopen('http://amp.pharm.mssm.edu/lincs-playground/index.php/s/F65mv0ZR81TwBIh/download')
CREEDS_data = json.load(resp)
CREEDS_df = pd.DataFrame(CREEDS_data)


# generate the up and down gene signatures
CREEDS_up_genes = {
    row['id']: row['up_genes']
    for row in CREEDS_data
}
CREEDS_down_genes = {
    row['id']: row['down_genes']
    for row in CREEDS_data
}


def get_geneset(df, indexer):
    df_ = df.loc[indexer, :]
    return list(df_[df_ == 1].index)

# L1000_up_genes = pd.read_csv(urllib.request.urlopen('https://amp.pharm.mssm.edu/lincs-playground/index.php/s/vvttTbwvoC653I0/download'))
# L1000_down_genes = pd.read_csv(urllib.request.urlopen('https://amp.pharm.mssm.edu/lincs-playground/index.php/s/j25mT86NsXQBfxI/download'))
# L1000_down_extract = pd.DataFrame(L1000_down_genes)
# L1000_up_extract = pd.DataFrame(L1000_up_genes)
### L1000 PRELOAD DATA DRUG CARD
#L1000_up_lookup.json
with urllib.request.urlopen('http://amp.pharm.mssm.edu/lincs-playground/index.php/s/ykWUcC6EwDer5cL/download') as f:
    L1000_up_lookup = json.load(f)

#L1000_down_lookup.json
with urllib.request.urlopen('http://amp.pharm.mssm.edu/lincs-playground/index.php/s/tkC8FqlWPkpkezZ/download') as f:
    L1000_down_lookup = json.load(f)

metadata = pd.read_csv(urllib.request.urlopen('https://amp.pharm.mssm.edu/lincs-playground/index.php/s/c2HKrkf8zCqAsfa/download'))
EMR_Drug_Names = EMR_data_df['Drug_Name']
unique_pert_ids = metadata['pert_id'].unique()
unique_drug_names = metadata['pert_desc'].unique()

#L1000_CREEDS_Similar_Drug_output.txt
with urllib.request.urlopen('http://amp.pharm.mssm.edu/lincs-playground/index.php/s/ro0KDMjUpFypEdJ/download') as fd:
    possible_drug_inputs = fd.read().decode().splitlines()

#possible_drug_inputs = set(unique_drug_names) & set(EMR_Drug_Names) # 28 possible drug inputs
drug_inputs = list(possible_drug_inputs)
drug_inputs_lowercase = [x.lower() for x in drug_inputs]
input_json = json.dumps(drug_inputs_lowercase)

    
with urllib.request.urlopen('http://amp.pharm.mssm.edu/lincs-playground/index.php/s/F65mv0ZR81TwBIh/download') as f:
    CREEDS_data = json.load(f)
CREEDS_GSE = {
    row['id']: [row['geo_id'], row["disease_name"]]
    for row in CREEDS_data
}

possible_disease_list = set(icd9_to_doid_final.Disease) 
#& set(EMR_data_df.Description) # will return {"Alzheimer's disease", "Barrett's esophagus", 'Dehydration', 'Sepsis'}
possible_disease_list = list(possible_disease_list) # this will be what the dropdown should display
while(" " in possible_disease_list) : 
    possible_disease_list.remove(" ") 
possible_disease_list_lowercase = [x.lower() for x in possible_disease_list]
disease_input_json = json.dumps(possible_disease_list_lowercase)


@app.route(ENDPOINT + "/")
def main():
    
    return render_template('index.html', inputs = input_json, disease_inputs = disease_input_json)

@app.route(ENDPOINT + "/search", methods=['POST'])
def return_doi():
    data = request.get_json(force=True)
    print(data)
    DOI = data['input'] # drug of interest
    print(DOI)
    return DOI

@app.route(ENDPOINT + '/drugs', methods=['POST'])
def display_drug_data():
    data = request.get_json(force=True)
    print(data)
    DrOI = data['input'] # drug of interest
    EMR_Drug_Names = EMR_data_df['Drug_Name'] # this will be the selection for the dropdown menu
    ## subset EMR data by the DOI and/or DrOI
    ## filter by DrOI need icd9 codes for proper conversion and query through CREEDS
    droi_search =EMR_data_df[EMR_data_df['Drug_Name'].apply(lambda s: bool(re.compile(DrOI, re.IGNORECASE).search(s)))]
    #top_disease_from_drug = EMR_top_disease_from_drug[0:5]
    #get rid of unnamed column with index 0
    droi_search = droi_search.drop(droi_search.columns[0], axis=1)
    EMR_display = droi_search[0:20]
    #rename columns
    droi_search.columns = ['ICD9', 'Description', 'Drug', 'Occurences']
    EMR_top_disease_from_drug = droi_search["ICD9"]
    ## build a datatable of all the ICD-9 CM diagnosis codes families (i.e no decimal points)
    EMR_top_disease_from_drug_df = pd.DataFrame(EMR_top_disease_from_drug, columns=['ICD9'])
    EMR_top_disease_from_drug_df['ICD9_wildcard'] = EMR_top_disease_from_drug_df['ICD9'].apply(lambda code: code.split('.')[0])
    icd9_to_doid_final['ICD9_wildcard'] = icd9_to_doid_final['ICD9'].apply(lambda code: str(code).split('.')[0])
    df_joined = pd.merge(
        left=EMR_top_disease_from_drug_df, left_on='ICD9_wildcard',
        right=icd9_to_doid_final, right_on='ICD9_wildcard',
        how='inner',
        suffixes=(
            '_left',
            '_right',
        )
    )
    from_drug_doids = df_joined["DOID"].unique()
    return render_template('drugs.html', name=DrOI, data=EMR_display.to_html(index=False))

@app.route(ENDPOINT + '/drug_specificity', methods=['POST'])
def drug_specificity():
    data = request.get_json(force=True)
    DOI = data['input']
    DOI_ICD9 = icd9_to_doid_final[icd9_to_doid_final.Disease.apply(lambda s: bool(re.compile(DOI, re.IGNORECASE).search(s)))]
    DOI_ICD9_codes = DOI_ICD9.ICD9
    # get the do_id from the DOI
    DOI_DOID_codes = DOI_ICD9.DOID
    # get the do_id from the DOI
    DOI_CREEDS_codes = DOI_ICD9.CREEDS_drug_id

    ##filter by DOI. Need to convert DOI to ICD9 first.
    icd9_to_doid_final_search = icd9_to_doid_final[icd9_to_doid_final["Disease"].apply(lambda s: bool(re.compile(DOI, re.IGNORECASE).search(s)))]
    icd9_to_doid_final_search1 = icd9_to_doid_final_search["ICD9"]
    ## rebuild the wildcard dataframe
    icd9_wildcard = pd.DataFrame(icd9_to_doid_final_search1, columns=['ICD9'])
    icd9_wildcard['ICD9_wildcard'] = icd9_wildcard['ICD9'].apply(lambda code: str(code).split('.')[0])
    icd9_wildcard.head()
    icd9_to_doid_final['ICD9_wildcard'] = icd9_to_doid_final['ICD9'].apply(lambda code: str(code).split('.')[0])
    #icd9_to_doid_final.head()
    ICD9_df_joined = pd.merge(
        left=icd9_wildcard, left_on='ICD9_wildcard',
        right=icd9_to_doid_final, right_on='ICD9_wildcard',
        how='inner',
        suffixes=(
            '_Manual',
            '_right',
        )
    )
    #ICD9_codes = str(int(ICD9_df_joined["ICD9_wildcard"].unique())) 
    ## generate an emr based on the ICD_9 codes extracted; can now extract the drug names as well
    #emr_sub = EMR_data_df[EMR_data_df['ICD9'].apply(lambda s: bool(re.compile(ICD9_codes, re.IGNORECASE).search(s)))]
    emr_sub = EMR_data_df[EMR_data_df["Description"].apply(lambda s: bool(re.compile(str(DOI), re.IGNORECASE).search(str(s))))]
    #emr_sub[0:10]
    emr_sub.reset_index(drop = True, inplace = True)
    emr_sub = []
    for a in ICD9_df_joined.ICD9_wildcard.unique():
        emr_sub1 = EMR_data_df[EMR_data_df["ICD9"].apply(lambda s: bool(re.compile(str(a), re.IGNORECASE).search(str(s))))]
        emr_sub.append(emr_sub1)
    emr_sub_df = pd.concat(emr_sub)
    #### L1000 integration
    # disease to drug conversion (disease input)
    total = emr_sub_df["Number_of_Occurences"].sum()
    emr_sub_df["Specificity_score"] = emr_sub_df["Number_of_Occurences"].div(total)
    top_drugs_from_disease = list(emr_sub_df.Drug_Name[0:10]) #take the top 5 drugs
    print(top_drugs_from_disease)
    columns = ['Drug_Name', 'ICD9', 'Number_of_Occurences', "Specificity_score"]
    top_drugs = emr_sub_df.iloc[0:20]
    top_drugs_columns = top_drugs[columns]
    specificity = emr_sub_df[columns]
    drugs_html = top_drugs_columns.to_html(index=False)
    return render_template('drug_specificity.html', drugData = drugs_html)

@app.route(ENDPOINT + '/emr_associations', methods=['GET', 'POST'])
def emr_associations():
    dod= request.args.get('drug')
    EMR_display_html = None 
    drugs_html = None
    if dod in drug_inputs_lowercase:
    ## Drug Data
        DrOI = dod
        EMR_Drug_Names = EMR_data_df['Drug_Name'] # this will be the selection for the dropdown menu
        ## subset EMR data by the DOI and/or DrOI
        ## filter by DrOI need icd9 codes for proper conversion and query through CREEDS
        droi_search =EMR_data_df[EMR_data_df['Drug_Name'].apply(lambda s: bool(re.compile(DrOI, re.IGNORECASE).search(s)))]
        #top_disease_from_drug = EMR_top_disease_from_drug[0:5]
        #get rid of unnamed column with index 0
        droi_search = droi_search.drop(droi_search.columns[0], axis=1)
        total = droi_search["Number_of_Occurences"].sum()
        droi_search["Specificity_score"] = droi_search["Number_of_Occurences"].div(total)
        droi_search.columns = ['ICD9', 'Description', 'Drug', 'Occurences', "Specificity"]
        EMR_display = droi_search[0:20]
        EMR_display_html = EMR_display.to_html(index=False)
        #rename columns
        EMR_top_disease_from_drug = droi_search["ICD9"]
        ## build a datatable of all the ICD-9 CM diagnosis codes families (i.e no decimal points)
        EMR_top_disease_from_drug_df = pd.DataFrame(EMR_top_disease_from_drug, columns=['ICD9'])
        EMR_top_disease_from_drug_df['ICD9_wildcard'] = EMR_top_disease_from_drug_df['ICD9'].apply(lambda code: code.split('.')[0])
        icd9_to_doid_final['ICD9_wildcard'] = icd9_to_doid_final['ICD9'].apply(lambda code: str(code).split('.')[0])
        df_joined = pd.merge(
            left=EMR_top_disease_from_drug_df, left_on='ICD9_wildcard',
            right=icd9_to_doid_final, right_on='ICD9_wildcard',
            how='inner',
            suffixes=(
                '_left',
                '_right',
            )
        )
        from_drug_doids = df_joined["DOID"].unique()
        
    if dod in possible_disease_list_lowercase:
        ## Disease Data
        DOI = dod
        DOI_ICD9 = icd9_to_doid_final[icd9_to_doid_final.Disease.apply(lambda s: bool(re.compile(DOI, re.IGNORECASE).search(s)))]
        DOI_ICD9_codes = DOI_ICD9.ICD9
        # get the do_id from the DOI
        DOI_DOID_codes = DOI_ICD9.DOID
        # get the do_id from the DOI
        DOI_CREEDS_codes = DOI_ICD9.CREEDS_drug_id

        ##filter by DOI. Need to convert DOI to ICD9 first.
        icd9_to_doid_final_search = icd9_to_doid_final[icd9_to_doid_final["Disease"].apply(lambda s: bool(re.compile(DOI, re.IGNORECASE).search(s)))]
        icd9_to_doid_final_search1 = icd9_to_doid_final_search["ICD9"]
        ## rebuild the wildcard dataframe
        icd9_wildcard = pd.DataFrame(icd9_to_doid_final_search1, columns=['ICD9'])
        icd9_wildcard['ICD9_wildcard'] = icd9_wildcard['ICD9'].apply(lambda code: str(code).split('.')[0])
        icd9_wildcard.head()
        icd9_to_doid_final['ICD9_wildcard'] = icd9_to_doid_final['ICD9'].apply(lambda code: str(code).split('.')[0])
        #icd9_to_doid_final.head()
        ICD9_df_joined = pd.merge(
            left=icd9_wildcard, left_on='ICD9_wildcard',
            right=icd9_to_doid_final, right_on='ICD9_wildcard',
            how='inner',
            suffixes=(
                '_Manual',
                '_right',
            )
        )
        #ICD9_codes = str(int(ICD9_df_joined["ICD9_wildcard"].unique())) 
        ## generate an emr based on the ICD_9 codes extracted; can now extract the drug names as well
        #emr_sub = EMR_data_df[EMR_data_df['ICD9'].apply(lambda s: bool(re.compile(ICD9_codes, re.IGNORECASE).search(s)))]
        emr_sub = EMR_data_df[EMR_data_df["Description"].apply(lambda s: bool(re.compile(str(DOI), re.IGNORECASE).search(str(s))))]
        #emr_sub[0:10]
        emr_sub.reset_index(drop = True, inplace = True)
        emr_sub = []
        for a in ICD9_df_joined.ICD9_wildcard.unique():
            emr_sub1 = EMR_data_df[EMR_data_df["ICD9"].apply(lambda s: bool(re.compile(str(a), re.IGNORECASE).search(str(s))))]
            emr_sub.append(emr_sub1)
        emr_sub_df = pd.concat(emr_sub)
        #### L1000 integration
        # disease to drug conversion (disease input)
        total = emr_sub_df["Number_of_Occurences"].sum()
        emr_sub_df["Specificity_score"] = emr_sub_df["Number_of_Occurences"].div(total)
        emr_sub_df.insert(loc=0, column="Disease", value=DOI.upper())
        top_drugs_from_disease = list(emr_sub_df.Drug_Name[0:10]) #take the top 5 drugs
        print(top_drugs_from_disease)
        columns = ['Disease','Drug_Name', 'ICD9', 'Number_of_Occurences', "Specificity_score"]
        top_drugs = emr_sub_df.iloc[0:20]
        top_drugs_columns = top_drugs[columns]
        top_drugs_columns.columns = ["Disease", "Drug", "ICD9", "Occurences", "Specificity"]
        drugs_html = top_drugs_columns.to_html(index=False)
    return render_template('emr_associations.html', drugData = EMR_display_html if EMR_display_html else drugs_html)
def process_blacklist(s):# EMR_display_html ? EMR_display_html : drugs_html
    blacklist = [
        # remove the classifications of the drugs
        re.compile(r'INJ', re.IGNORECASE),
        re.compile(r'CAP', re.IGNORECASE),
        re.compile(r'\d+', re.IGNORECASE),
        
        # remove drugs that aren't in the L1000
        re.compile(r'SODIUM', re.IGNORECASE),
        re.compile(r'HEPATITIS', re.IGNORECASE),
        re.compile(r'HEPARIN', re.IGNORECASE),
        re.compile(r'CALCIUM', re.IGNORECASE),
        re.compile(r'ZZ', re.IGNORECASE),
        
    ]
    for b in blacklist:
        s = re.sub(b, '', s)
    return s.strip()

@app.route(ENDPOINT + '/L1000', methods=['POST'])
def display_L1000():
    data = request.get_json(force=True)
    #print(data)
    DrOI = data['input']
    #print(DrOI)
    ### try the input
    # DrOI = "DIGOXIN" # drug of interest. MAKE SURE TO SELECT THIS FROM "possible_drug_inputs"
####

    DrOI_df = metadata[metadata["pert_desc"] == DrOI]
    DrOI_pert_ids = list(DrOI_df["pert_id"])
    DrOI_up_signatures = {k: L1000_up_lookup[k] for k in (DrOI_pert_ids)}
    DrOI_up_no_perts = dict(*DrOI_up_signatures.values())
    DrOI_up_drug_sigs = list(DrOI_up_no_perts.keys())
    DrOI_down_signatures = {k: L1000_down_lookup[k] for k in (DrOI_pert_ids)}
    DrOI_down_no_perts = dict(*DrOI_down_signatures.values())
    DrOI_down_drug_sigs = list(DrOI_down_no_perts.keys())
    DrOI_all_sigs = set(DrOI_up_drug_sigs) & set (DrOI_down_drug_sigs)
    DrOI_all_sigs_df = pd.DataFrame(set(DrOI_up_drug_sigs) & set (DrOI_down_drug_sigs))
    L1000_all_json = {}
    for a in DrOI_all_sigs:
        print(a)
        L1000_up_json_file = DrOI_up_no_perts[a]
        L1000_down_json_file = DrOI_down_no_perts[a]
        L1000_genes = {"up_genes": L1000_up_json_file, "down_genes": L1000_down_json_file}
        L1000_json= {a: L1000_genes}
        L1000_all_json.update(L1000_json)
    L1000_gene_json = json.dumps(L1000_all_json)

    return render_template('L1000.html', L1000_sig_ids = DrOI_all_sigs_df.to_json(), L1000_genes = L1000_gene_json)
        
@app.route(ENDPOINT + '/L1000_diseases', methods=['POST'])
def l1000_diseases():
    data = request.get_json(force=True)
    DOI = data['input']
    ### DISEASE --> ICD9 --> DOI
# get the ICD9 from the DOI
    DOI_ICD9 = icd9_to_doid_final[icd9_to_doid_final.Disease.apply(lambda s: bool(re.compile(DOI, re.IGNORECASE).search(s)))]
    DOI_ICD9_codes = DOI_ICD9.ICD9

    # get the do_id from the DOI
    DOI_DOID_codes = DOI_ICD9.DOID

    # get the do_id from the DOI
    DOI_CREEDS_codes = DOI_ICD9.CREEDS_drug_id
        
    ##filter by DOI. Need to convert DOI to ICD9 first.
    icd9_to_doid_final_search = icd9_to_doid_final[icd9_to_doid_final["Disease"].apply(lambda s: bool(re.compile(DOI, re.IGNORECASE).search(s)))]
    icd9_to_doid_final_search1 = icd9_to_doid_final_search["ICD9"]

    ## rebuild the wildcard dataframe
    icd9_wildcard = pd.DataFrame(icd9_to_doid_final_search1, columns=['ICD9'])
    icd9_wildcard['ICD9_wildcard'] = icd9_wildcard['ICD9'].apply(lambda code: str(code).split('.')[0])
    icd9_wildcard.head()

    icd9_to_doid_final['ICD9_wildcard'] = icd9_to_doid_final['ICD9'].apply(lambda code: str(code).split('.')[0])
    ICD9_df_joined = pd.merge(
        left=icd9_wildcard, left_on='ICD9_wildcard',
        right=icd9_to_doid_final, right_on='ICD9_wildcard',
        how='inner',
        suffixes=(
            '_Manual',
            '_right',
        )
    )

    emr_sub = EMR_data_df[EMR_data_df["Description"].apply(lambda s: bool(re.compile(str(DOI), re.IGNORECASE).search(str(s))))]
    emr_sub.reset_index(drop = True, inplace = True)

    emr_sub = []
    for a in ICD9_df_joined.ICD9_wildcard.unique():
        emr_sub1 = EMR_data_df[EMR_data_df["ICD9"].apply(lambda s: bool(re.compile(str(a), re.IGNORECASE).search(str(s))))]
        emr_sub.append(emr_sub1)
    emr_sub_df = pd.concat(emr_sub)
    top_drugs_from_disease = list(emr_sub_df.Drug_Name[0:15]) #take the top 5 drugs
    top_drugs_from_disease


    single_word_drugs = []
    for i in top_drugs_from_disease:
        j = str(i)
        splitted = j.split()
        first_word = splitted[0]
        single_word_drugs.append(first_word) 
    single_word_drugs = list(set(single_word_drugs))
    single_word_drugs_list = list(pd.Series(single_word_drugs).map(process_blacklist))

    L1000FWD_URL = 'http://amp.pharm.mssm.edu/L1000FWD/'


    L1000_reverse_drugs_store = []
    L1000_reverse_pertids_store = []

    for a in single_word_drugs_list:
        query_string = a
        L1000_reverse_drug_response = requests.get(L1000FWD_URL + 'synonyms/' + query_string)
        if L1000_reverse_drug_response.status_code == 200:
            #pprint(L1000_reverse_drug_response.json())
            L1000_reverse_significant_query = L1000_reverse_drug_response.json()
            if len(L1000_reverse_significant_query) > 0:
                #json.dump(L1000_reverse_drug_response.json(), open(query_string + '_L1000_reverse_drug_query.json', 'w'), indent=4)
                L1000_reverse_significant_query = L1000_reverse_drug_response.json()
                L1000_reverse_significant_query_df = pd.DataFrame(L1000_reverse_significant_query)
                L1000_reverse_pertids_store.append(list(L1000_reverse_significant_query_df["pert_id"]))
                L1000_reverse_drugs_store.append(a)
                print("Found significant L1000 drug signatures for " + query_string)

                
            else:
                print("No significant L1000 drug signatures for " + query_string)          
    
        
    L1000_reverse_pertids_flat = []
    for sublist in L1000_reverse_pertids_store:
        for item in sublist:
            L1000_reverse_pertids_flat.append(item) 

    L1000_reverse_pertids_flat = set(list(L1000_reverse_pertids_flat)) & set(list(metadata["pert_id"])) & set(list(L1000_up_lookup.keys()))

    DrOI_disease_up_signatures = {k: L1000_up_lookup[k] for k in (L1000_reverse_pertids_flat)}
    DrOI_disease_up_no_perts = {k: v for d in DrOI_disease_up_signatures.values() for k, v in d.items()}
    DrOI_disease_up_drug_sigs = list(DrOI_disease_up_no_perts.keys())

    DrOI_disease_down_signatures = {k: L1000_down_lookup[k] for k in (L1000_reverse_pertids_flat)}
    DrOI_disease_down_no_perts = {k: v for d in DrOI_disease_down_signatures.values() for k, v in d.items()}
    DrOI_disease_down_drug_sigs = list(DrOI_disease_down_no_perts.keys())

    DrOI_disease_all_sigs = set(DrOI_disease_up_drug_sigs) & set(DrOI_disease_down_drug_sigs)
    
    L1000_genes = {}
    L1000_sigs = {}
    L1000_json = {}

    for a in DrOI_disease_all_sigs:
        L1000_up_json_file = DrOI_disease_up_no_perts[a]
        L1000_down_json_file = DrOI_disease_down_no_perts[a]
        L1000_genes = {"up_genes": L1000_up_json_file, "down_genes": L1000_down_json_file}
        L1000_sigs= {a: L1000_genes}
        L1000_json.update(L1000_sigs)
    L1000_gene_json = json.dumps(L1000_json)
    return render_template('L1000_diseases.html', L1000_json = L1000_gene_json)

@app.route(ENDPOINT + "/geneshot", methods=['POST'])
def display_Geneshot():
    data = request.get_json(force=True)
    #print(data)
    DrOI = data['input']
    #print(DrOI)
    GENESHOT_URL = 'http://amp.pharm.mssm.edu/geneshot/api'
    query_string = '/search/auto/%s'
    search_term = DrOI # this will be the user input 
    geneshot_response = requests.get(
        GENESHOT_URL + query_string % (search_term)
    )
    if not geneshot_response.ok:
        raise Exception('Error during query')

    geneshot_data = json.loads(geneshot_response.text)
    #print(geneshot_data)
    geneshot_gene_df = geneshot_data["gene_count"]
    geneshot_gene_list = list(geneshot_gene_df.keys()) # this extracts the genes from the json. We can then resend this through the geneshot api
    geneshot_gene_list_commas = ",".join(geneshot_gene_list) # can save this as a csv. 

    query_string = '/associate/%s/%s'
    similarity_matrix = 'coexpression' # we can make this dynamic. Parameters: (generif, tagger, autorif, coexpression, enrichr)
    gene_symbols = geneshot_gene_list_commas

    coexpression_response = requests.get(
        GENESHOT_URL + query_string % (similarity_matrix, gene_symbols)
    )
    if not coexpression_response.ok:
        raise Exception('Error during query')

    coexpression_data = json.loads(coexpression_response.text) # this will be the Archs4 coexpression json they can download


    query_string = '/associate/%s/%s'
    similarity_matrix = 'generif' # we can make this dynamic. Parameters: (generif, tagger, autorif, coexpression, enrichr)
    gene_symbols = geneshot_gene_list_commas

    generif_response = requests.get(
        GENESHOT_URL + query_string % (similarity_matrix, gene_symbols)
    )
    if not generif_response.ok:
        raise Exception('Error during query')

    generif_data = json.loads(generif_response.text) # this will be the GENERIF coexpression json they can download

#generif_data

    query_string = '/associate/%s/%s'
    similarity_matrix = 'tagger' # we can make this dynamic. Parameters: (generif, tagger, autorif, coexpression, enrichr)
    gene_symbols = geneshot_gene_list_commas

    tagger_response = requests.get(
        GENESHOT_URL + query_string % (similarity_matrix, gene_symbols)
    )
    if not tagger_response.ok:
        raise Exception('Error during query')

    tagger_data = json.loads(tagger_response.text) # this will be the TAGGER coexpression json they can download
    #tagger_data



    query_string = '/associate/%s/%s'
    similarity_matrix = 'tagger' # we can make this dynamic. Parameters: (generif, tagger, autorif, coexpression, enrichr)
    gene_symbols = geneshot_gene_list_commas

    autorif_response = requests.get(
        GENESHOT_URL + query_string % (similarity_matrix, gene_symbols)
    )
    if not autorif_response.ok:
        raise Exception('Error during query')

    autorif_data = json.loads(autorif_response.text) # this will be the coexpression json they can download
    #autorif_data

    query_string = '/associate/%s/%s'
    similarity_matrix = 'tagger' # we can make this dynamic. Parameters: (generif, tagger, autorif, coexpression, enrichr)
    gene_symbols = geneshot_gene_list_commas

    enrichr_response = requests.get(
        GENESHOT_URL + query_string % (similarity_matrix, gene_symbols)
    )
    if not enrichr_response.ok:
        raise Exception('Error during query')

    enrichr_data = json.loads(enrichr_response.text) # this will be the coexpression json they can download
    #enrichr_data

    return render_template('geneshot.html', data = geneshot_data, archs4 = coexpression_data, generif = generif_data, tagger = tagger_data, autorif = autorif_data, enrichr = enrichr_data, drug = data)

@app.route(ENDPOINT + "/creeds_rx_dx", methods=['POST'])
def creeds_rx_dx():
    data = request.get_json(force=True)
    DOI = data['input'] # disease of interest. CAN TAKE FROM posssible_disease_list FOR NOW
    ####
    #possible_diseases = EMR_data_df["Description"] #possible diseases from the Sinai EMR

    DOI_ICD9 = icd9_to_doid_final[icd9_to_doid_final.Disease.apply(lambda s: bool(re.compile(DOI, re.IGNORECASE).search(s)))]
    DOI_ICD9_codes = DOI_ICD9.ICD9
    # get the do_id from the DOI
    DOI_DOID_codes = DOI_ICD9.DOID
    # get the do_id from the DOI
    DOI_CREEDS_codes = DOI_ICD9.CREEDS_drug_id
        
    ##filter by DOI. Need to convert DOI to ICD9 first.
    icd9_to_doid_final_search = icd9_to_doid_final[icd9_to_doid_final["Disease"].apply(lambda s: bool(re.compile(DOI, re.IGNORECASE).search(s)))]
    icd9_to_doid_final_search1 = icd9_to_doid_final_search["ICD9"]
    ## rebuild the wildcard dataframe
    icd9_wildcard = pd.DataFrame(icd9_to_doid_final_search1, columns=['ICD9'])
    icd9_wildcard['ICD9_wildcard'] = icd9_wildcard['ICD9'].apply(lambda code: str(code).split('.')[0])
    icd9_wildcard.head()
    icd9_to_doid_final['ICD9_wildcard'] = icd9_to_doid_final['ICD9'].apply(lambda code: str(code).split('.')[0])
    #icd9_to_doid_final.head()
    ICD9_df_joined = pd.merge(
        left=icd9_wildcard, left_on='ICD9_wildcard',
        right=icd9_to_doid_final, right_on='ICD9_wildcard',
        how='inner',
        suffixes=(
            '_Manual',
            '_right',
        )
    )
    #ICD9_codes = str(int(ICD9_df_joined["ICD9_wildcard"].unique())) 
    ## generate an emr based on the ICD_9 codes extracted; can now extract the drug names as well
    #emr_sub = EMR_data_df[EMR_data_df['ICD9'].apply(lambda s: bool(re.compile(ICD9_codes, re.IGNORECASE).search(s)))]
    emr_sub = EMR_data_df[EMR_data_df["Description"].apply(lambda s: bool(re.compile(str(DOI), re.IGNORECASE).search(str(s))))]
    #emr_sub[0:10]
    emr_sub.reset_index(drop = True, inplace = True)
    emr_sub = []
    for a in ICD9_df_joined.ICD9_wildcard.unique():
        emr_sub1 = EMR_data_df[EMR_data_df["ICD9"].apply(lambda s: bool(re.compile(str(a), re.IGNORECASE).search(str(s))))]
        emr_sub.append(emr_sub1)
    emr_sub_df = pd.concat(emr_sub)
    #### L1000 integration
    # disease to drug conversion (disease input)
    top_drugs_from_disease = list(emr_sub_df.Drug_Name[0:10]) #take the top 5 drugs
    metadata = pd.read_csv(urllib.request.urlopen('https://amp.pharm.mssm.edu/lincs-playground/index.php/s/c2HKrkf8zCqAsfa/download'))
    metadata ## same as LINC1000h5.row_metadata_df
    #metadata
    for a in top_drugs_from_disease:
        print(a)
        meta_doi = metadata[metadata["pert_desc"].apply(lambda s: bool(re.compile(str(a), re.IGNORECASE).search(str(s))))]
    meta_doi
    meta_doi_ids = meta_doi.rid
    query = list(meta_doi_ids)
    #print(query)
    # disease to drug conversion (disease input)
    top_drugs_from_disease = list(emr_sub_df.Drug_Name[0:20]) #take the top 5 drugs
    top_drugs_from_disease
    single_word_drugs = []
    for i in top_drugs_from_disease:
        j = str(i)
        splitted = j.split()
        first_word = splitted[0]
    # print(first_word)
        single_word_drugs.append(first_word) 
    #print(single_word_drugs)
    single_word_drugs = list(set(single_word_drugs))

    # Generate a blacklist process
    single_word_drugs_list = list(pd.Series(single_word_drugs).map(process_blacklist))



        ### CREEDS DRUG CARD 
    while("" in single_word_drugs_list):
        single_word_drugs_list.remove("")
    
    single_word_drugs_list
    CREEDS_URL = 'http://amp.pharm.mssm.edu/CREEDS/'
    CREEDS_drug_from_disease_up_genes = []
    CREEDS_drug_from_disease_down_genes = []
    up_signature_names = []
    down_signature_names = []
    CREEDS_drug_ids_store = []
    for drug_name in single_word_drugs_list:
        CREEEDS_Drug_response = requests.get(CREEDS_URL + 'search', params={'q':str(drug_name)})
        if CREEEDS_Drug_response.status_code == 200:
            CREEDS_drug_output_df = pd.DataFrame(CREEEDS_Drug_response.json())
            
            if len(CREEDS_drug_output_df) > 0:
                CREEDS_drug_output_ids = list(CREEDS_drug_output_df["id"])
                CREEDS_drug_ids_store.append(CREEDS_drug_output_ids)
                CREEDS_drug_output_ids_df = pd.DataFrame(list(itertools.chain(*CREEDS_drug_ids_store)))
                
                print("CREEDS IDs found for " + drug_name)
                for outputid in CREEDS_drug_output_ids:
                    CREEDS_drug_sigs_response = requests.get(CREEDS_URL + 'api', params={'id':outputid})
                    if CREEDS_drug_sigs_response.status_code == 200:
                        CREEDS_drug_sigs_response_json = CREEDS_drug_sigs_response.json()

                        ## up genes
                        CREEDS_drug_sigs_up_genes = CREEDS_drug_sigs_response_json['up_genes']
                        CREEDS_drug_sigs_up_genes_df = pd.DataFrame(CREEDS_drug_sigs_up_genes) # this is the up genes dataframe
                        filename1 = (drug_name + "_CREEDS_drug_sig_up_genes.csv")
                        up_signature_names_temp = (DOI + "_" + 
                                                drug_name + "_" + 
                                                CREEDS_drug_sigs_response_json["geo_id"] + "_" +
                                                CREEDS_drug_sigs_response_json["id"]
                                                )
                        up_signature_names.append(up_signature_names_temp)
    #                     up_signature_names = pd.DataFrame(up_signature_names)
                        up_signatures_df = pd.DataFrame(up_signature_names)
                        print(up_signature_names)
                        #CREEDS_drug_sigs_up_genes_df.to_csv(filename1) # this saves the df as a csv
                        ## down genes
                        CREEDS_drug_sigs_down_genes = CREEDS_drug_sigs_response_json['down_genes']
                        CREEDS_drug_sigs_down_genes_df = pd.DataFrame(CREEDS_drug_sigs_down_genes)# this is the down genes dataframe
                        filename2 = (drug_name + "_CREEDS_drug_sig_down_genes.csv")
                        down_signature_names_temp = (DOI + "_" + 
                                                drug_name + "_" +
                                                CREEDS_drug_sigs_response_json["geo_id"]+ "_" +
                                                CREEDS_drug_sigs_response_json["id"]
                                                )
                        print(down_signature_names_temp)
                        down_signature_names.append(down_signature_names_temp)
                        down_signatures_df = pd.DataFrame(down_signature_names)
                        
                        #CREEDS_drug_sigs_down_genes_df.to_csv(filename2)
                        #CREEDS_drug_sigs_down_genes = CREEDS_drug_sigs_response_json['down_genes'] # this saves the df as a csv
                        ## json propagation
                        #pprint(response.json())
                        #json.dump(response.json(), open(a + '_CREEDS_Drug_sig.json', 'w'), indent=4) # if the user wants the entire json, they can download this
                        CREEDS_drug_from_disease_up_genes.append(CREEDS_drug_sigs_up_genes)
                        CREEDS_drug_from_disease_down_genes.append(CREEDS_drug_sigs_down_genes_df)
            else:
                print ("No CREEDS IDs found for " + drug_name)
    return render_template('creeds_rx_dx.html', CREEDS_drug_ids = CREEDS_drug_output_ids_df.to_json(), CREEDS_up_names = up_signatures_df.to_json(), CREEDS_down_names = down_signatures_df.to_json())

def return_creeds_rx_dx_up_df(sig_id):
    CREEDS_URL = 'http://amp.pharm.mssm.edu/CREEDS/'
    CREEDS_drug_sigs_response = requests.get(CREEDS_URL + 'api', params={'id':str(sig_id)})
    if CREEDS_drug_sigs_response.status_code == 200:
        CREEDS_drug_sigs_response_json = CREEDS_drug_sigs_response.json()

        ## up genes
        CREEDS_drug_sigs_up_genes = CREEDS_drug_sigs_response_json['up_genes']
        CREEDS_drug_sigs_up_genes_df = pd.DataFrame(CREEDS_drug_sigs_up_genes) # this is the up genes dataframe
        CREEDS_drug_sigs_up_genes_df.columns = ['Gene', 'Score']
        #filename1 = (drug_name + "_CREEDS_drug_sig_up_genes.csv")
        # up_signature_names_temp = (DOI + "_" + 
        #                         drug_name + "_" + 
        #                         CREEDS_drug_sigs_response_json["geo_id"] + "_" +
        #                         CREEDS_drug_sigs_response_json["id"]
        #                         )
        # up_signature_names.append(up_signature_names_temp)
        # up_signature_names = pd.DataFrame(up_signature_names)
        #CREEDS_drug_sigs_up_genes_df.to_csv(filename1) # this saves the df as a csv
        ## down genes
        CREEDS_drug_sigs_down_genes = CREEDS_drug_sigs_response_json['down_genes']
        CREEDS_drug_sigs_down_genes_df = pd.DataFrame(CREEDS_drug_sigs_down_genes)# this is the down genes dataframe
        #filename2 = (drug_name + "_CREEDS_drug_sig_down_genes.csv")
        #down_signature_names_temp = (DOI + "_" + 
                                #drug_name + "_" +
                                #CREEDS_drug_sigs_response_json["geo_id"]+ "_" +
                                #CREEDS_drug_sigs_response_json["id"]
                                #)
        #print(down_signature_names_temp)
        # down_signature_names.append(down_signature_names_temp)
        # down_signature_names = pd.DataFrame(down_signature_names)
        
        #CREEDS_drug_sigs_down_genes_df.to_csv(filename2)
        #CREEDS_drug_sigs_down_genes = CREEDS_drug_sigs_response_json['down_genes'] # this saves the df as a csv
        ## json propagation
        #pprint(response.json())
        #json.dump(response.json(), open(a + '_CREEDS_Drug_sig.json', 'w'), indent=4) # if the user wants the entire json, they can download this
        # CREEDS_drug_from_disease_up_genes.append(CREEDS_drug_sigs_up_genes)
        # CREEDS_drug_from_disease_down_genes.append(CREEDS_drug_sigs_down_genes_df)
    else:
        print ("No CREEDS IDs found for "+ sig_id)
    return CREEDS_drug_sigs_up_genes_df

def return_creeds_rx_dx_down_df(sig_id):
    CREEDS_URL = 'http://amp.pharm.mssm.edu/CREEDS/'
    CREEDS_drug_sigs_response = requests.get(CREEDS_URL + 'api', params={'id':str(sig_id)})
    if CREEDS_drug_sigs_response.status_code == 200:
        CREEDS_drug_sigs_response_json = CREEDS_drug_sigs_response.json()

        ## up genes
        CREEDS_drug_sigs_up_genes = CREEDS_drug_sigs_response_json['up_genes']
        CREEDS_drug_sigs_up_genes_df = pd.DataFrame(CREEDS_drug_sigs_up_genes) # this is the up genes dataframe
        # filename1 = (drug_name + "_CREEDS_drug_sig_up_genes.csv")
        # up_signature_names_temp = (DOI + "_" + 
        #                         drug_name + "_" + 
        #                         CREEDS_drug_sigs_response_json["geo_id"] + "_" +
        #                         CREEDS_drug_sigs_response_json["id"]
        #                         )
        # up_signature_names.append(up_signature_names_temp)
        # up_signature_names = pd.DataFrame(up_signature_names)
        #CREEDS_drug_sigs_up_genes_df.to_csv(filename1) # this saves the df as a csv
        ## down genes
        CREEDS_drug_sigs_down_genes = CREEDS_drug_sigs_response_json['down_genes']
        CREEDS_drug_sigs_down_genes_df = pd.DataFrame(CREEDS_drug_sigs_down_genes)# this is the down genes dataframe
        CREEDS_drug_sigs_down_genes_df.columns = ['Gene', 'Score']
        
        # filename2 = (drug_name + "_CREEDS_drug_sig_down_genes.csv")
        # down_signature_names_temp = (DOI + "_" + 
        #                         drug_name + "_" +
        #                         CREEDS_drug_sigs_response_json["geo_id"]+ "_" +
        #                         CREEDS_drug_sigs_response_json["id"]
        #                         )
        # print(down_signature_names_temp)
        # down_signature_names.append(down_signature_names_temp)
        # down_signature_names = pd.DataFrame(down_signature_names)
        
        #CREEDS_drug_sigs_down_genes_df.to_csv(filename2)
        #CREEDS_drug_sigs_down_genes = CREEDS_drug_sigs_response_json['down_genes'] # this saves the df as a csv
        ## json propagation
        #pprint(response.json())
        #json.dump(response.json(), open(a + '_CREEDS_Drug_sig.json', 'w'), indent=4) # if the user wants the entire json, they can download this
        # CREEDS_drug_from_disease_up_genes.append(CREEDS_drug_sigs_up_genes)
        # CREEDS_drug_from_disease_down_genes.append(CREEDS_drug_sigs_down_genes_df)
    else:
        print ("No CREEDS IDs found for " + sig_id)
    return CREEDS_drug_sigs_down_genes_df

@app.route(ENDPOINT + "/creeds_dx_dx", methods=['POST'])
def creeds_diseases_diseases():
    data = request.get_json(force=True)
    DOI = data['input']
    CREEDS_disease = CREEDS_df[CREEDS_df["disease_name"] == DOI]
    CREEDS_disease['full_name'] = CREEDS_disease[["disease_name", "geo_id", "organism"]].apply(lambda x: ' '.join(x), axis=1)
    CREEDS_full_name = list(CREEDS_disease["full_name"])
    CREEDS_full_name_df = pd.DataFrame(CREEDS_full_name)
    CREEDS_full_name_df.columns = ['full_name']
    CREEDS_disease_ids = list(CREEDS_disease["id"]) # all the signatures to print out
    CREEDS_disease_ids_df = pd.DataFrame(CREEDS_disease_ids)
    CREEDS_disease_ids_df.columns = ["id"]

   
    return render_template('creeds_dx_dx.html', creeds_full_name = CREEDS_full_name_df.to_json(), creeds_dx_json =  CREEDS_disease_ids_df.to_json())

def return_creeds_dx_dx_up_df(sig_id):
    CREEDS_up_gene_sigs = pd.DataFrame(CREEDS_up_genes[str(sig_id)])
    CREEDS_up_gene_sigs.columns = ["Gene", "Score"]
    # CREEDS_URL = 'http://amp.pharm.mssm.edu/CREEDS/'
    # CREEDS_dz_id_json = requests.get(CREEDS_URL + 'api', params={'id': 'dz:503'})
    # CREEDS_sig_dz_json = CREEDS_dz_id_json.json()
    # CREEDS_disease_sig_up_genes = CREEDS_sig_dz_json['up_genes']
    # CREEDS_disease_sig_up_genes_df = pd.DataFrame(CREEDS_disease_sig_up_genes)
    #  # this is the up genes
    # CREEDS_disease_sig_up_genes_df.c 
    return CREEDS_up_gene_sigs

def return_creeds_dx_dx_down_df(sig_id):
    CREEDS_down_gene_sigs = pd.DataFrame(CREEDS_down_genes[str(sig_id)])
    CREEDS_down_gene_sigs.columns = ["Gene", "Score"]
    # CREEDS_URL = 'http://amp.pharm.mssm.edu/CREEDS/'
    # CREEDS_dz_id_json = requests.get(CREEDS_URL + 'api', params={'id': 'dz:503'})
    # CREEDS_sig_dz_json = CREEDS_dz_id_json.json()
    # CREEDS_disease_sig_down_genes = CREEDS_sig_dz_json['down_genes']
    # CREEDS_disease_sig_down_genes_df = pd.DataFrame(CREEDS_disease_sig_down_genes)
    return CREEDS_down_gene_sigs

@app.route(ENDPOINT + "/creeds_drugs", methods=['POST'])
def display_creeds_drugs_drug_input():
    data = request.get_json(force=True)
    DrOI = data['input']
    CREEDS_URL = 'http://amp.pharm.mssm.edu/CREEDS/'
    CREEDS_Drug_response = requests.get(CREEDS_URL + 'search', params={'q':DrOI})
    if CREEDS_Drug_response.status_code == 200:
    #pprint(CREEEDS_Drug_response.json())
    #json.dump(CREEEDS_Drug_response.json(), open(DrOI + '_api1_result.json', 'w'), indent=4)
        CREEDS_drug_output_df = pd.DataFrame(CREEDS_Drug_response.json())
        CREEDS_drug_output_df['name'] = DrOI
        CREEDS_drug_output_df['full_name'] = CREEDS_drug_output_df[['name', 'geo_id', 'organism']].apply(lambda x: ' '.join(x), axis=1)
        CREEDS_name_df = pd.DataFrame(CREEDS_drug_output_df['full_name'])
        CREEDS_drug_output_ids = pd.DataFrame(CREEDS_drug_output_df['id'])

    return render_template('creeds_drugs.html', creeds_names = CREEDS_name_df.to_json(), creeds_ids = CREEDS_drug_output_ids.to_json())


@app.route(ENDPOINT + "/creeds_diseases", methods=['POST'])
def display_creeds_diseases_drug_input():
    data = request.get_json(force=True)
    #print(data)
    DrOI = data['input']
    ### CREEDS DISEASE CARD (DRUG INPUT)
    #disease signatures json
#     with urllib.request.urlopen('https://amp.pharm.mssm.edu/lincs-playground/index.php/s/F65mv0ZR81TwBIh') as f:
#         CREEDS_data = json.load(f)
# # RETURNS THE do_id, geo_id, and disease name in a dictionary
#     ### CREEDS DISEASE CARD (DRUG INPUT)
#     CREEDS_df = pd.DataFrame(CREEDS_data)
#     CREEDS_up_genes = {
#     row['id']: row['up_genes']
#     for row in CREEDS_data
#     }
#     CREEDS_down_genes = {
#         row['id']: row['down_genes']
#         for row in CREEDS_data
#     }

#     CREEDS_disease = CREEDS_df[CREEDS_df["disease_name"] == DOI]
#     CREEDS_disease_ids = list(CREEDS_disease["id"])
    
# RETURNS THE do_id, geo_id, and disease name in a dictionary
    CREEDS_GSE = {
    row['id']: [row['geo_id'], row["disease_name"]]
    for row in CREEDS_data
    }

## filter by DrOI need icd9 codes for proper conversion and query through CREEDS
    droi_search =EMR_data_df[EMR_data_df['Drug_Name'].apply(lambda s: bool(re.compile(DrOI, re.IGNORECASE).search(s)))]
    droi_search_top5 = droi_search[0:5]
    EMR_top_disease_from_drug = droi_search_top5["ICD9"]
    #top_disease_from_drug = EMR_top_disease_from_drug[0:5]
    ## build a datatable of all the ICD-9 CM diagnosis codes families (i.e no decimal points)
    EMR_top_disease_from_drug_df = pd.DataFrame(EMR_top_disease_from_drug, columns=['ICD9'])
    EMR_top_disease_from_drug_df['ICD9_wildcard'] = EMR_top_disease_from_drug_df['ICD9'].apply(lambda code: code.split('.')[0])
    #EMR_top_disease_from_drug_df.head()
    icd9_to_doid_final['ICD9_wildcard'] = icd9_to_doid_final['ICD9'].apply(lambda code: str(code).split('.')[0])
    #icd9_to_doid_final.head()
    df_joined = pd.merge(
        left=EMR_top_disease_from_drug_df, left_on='ICD9_wildcard',
        right=icd9_to_doid_final, right_on='ICD9_wildcard',
        how='inner',
        suffixes=(
            '_left',
            '_right',
        )
    )
    CREEDS_drug_ids = pd.DataFrame(set(df_joined.CREEDS_drug_id))
    CREEDS_drug_ids.columns = ["id"]
    CREEDS_drug_ids_list = list(set(df_joined.CREEDS_drug_id))
    #CREEDS_GSE.keys()
    #CREEDS_drug_ids_list
    CREEDS_Drug_Final = dict((k, CREEDS_GSE[k]) for k in CREEDS_drug_ids_list)
    CREEDS_drug_final_df = pd.DataFrame(CREEDS_Drug_Final).T
    CREEDS_drug_final_df.columns = ["GSE_ID", "DISEASE"]
    CREEDS_drug_final_df["drug"] = DrOI
    CREEDS_drug_final_df["full_name"] = CREEDS_drug_final_df[["drug", "GSE_ID", "DISEASE"]].apply(lambda x: ' '.join(x), axis=1)
    #CREEDS_drug_final_df # DISPLAY THIS DATAFRAME
    ### CREEDS DISEASE CARD FROM DRUG INPUT API
    # CREEDS_drug_final_diseases = CREEDS_drug_final_df.DISEASE
    # CREEDS_drug_final_GSE_ID = CREEDS_drug_final_df.GSE_ID
    # ## CREEDS DISEASE CARD FROM DISEASE QUERY 
    # ​
    # loop_iteration = np.arange(0, len(CREEDS_drug_final_diseases))
    # loop_iteration = list(loop_iteration)


    return render_template('creeds_diseases.html', creeds_drug_ids = CREEDS_drug_ids.to_json(), creeds_drug_fullnames = CREEDS_drug_final_df.to_json())

def run_X2K(input_genes, options={}):
    # Open HTTP connection
    conn = http.client.HTTPConnection("amp.pharm.mssm.edu")

    # Set default options
    default_options = {'text-genes': '\n'.join(input_genes),
                       'included_organisms': 'both',
                       'TF-target gene background database used for enrichment': 'ChEA & ENCODE Consensus',
                       'sort transcription factors by': 'p-value',
                       'min_network_size': 10,
                       'number of top TFs': 10,
                       'path_length': 2,
                       'min_number_of_articles_supporting_interaction': 0,
                       'max_number_of_interactions_per_protein': 200,
                       'max_number_of_interactions_per_article': 100,
                       'enable_BioGRID': True,
                       'enable_IntAct': True,
                       'enable_MINT': True,
                       'enable_ppid': True,
                       'enable_Stelzl': True,
                       'kinase interactions to include': 'kea 2018',
                       'sort kinases by': 'p-value'}

    # Update options
    for key, value in options.items():
        if key in default_options.keys() and key != 'text-genes':
            default_options.update({key: value})

    # Get payload
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    payload = ''.join(
        ['--' + boundary + '\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n'.format(**locals())
         for key, value in default_options.items()]) + '--' + boundary + '--'

    # Get Headers
    headers = {
        'content-type': "multipart/form-data; boundary=" + boundary,
        'cache-control': "no-cache",
    }

    # Initialize connection
    conn.request("POST", "/X2K/api", payload, headers)

    # Get response
    res = conn.getresponse()

    # Read response
    data = res.read().decode('utf-8')

    # Convert to dictionary
    x2k_results = {key: json.loads(value) if key != 'input' else value for key, value in json.loads(data).items()}

    # Clean results
    x2k_results['ChEA'] = x2k_results['ChEA']['tfs']
    x2k_results['G2N'] = x2k_results['G2N']['network']
    x2k_results['KEA'] = x2k_results['KEA']['kinases']
    x2k_results['X2K'] = x2k_results['X2K']['network']

    # Return results
    return x2k_results


@app.route(ENDPOINT + '/x2kl1000', methods=['POST'])
def display_x2kl1000():
### try the input
    data = request.get_json(force=True)
    #print(data)
    DrOI = data['input']
    # drug of interest. MAKE SURE TO SELECT THIS FROM "possible_drug_inputs"
####
    DrOI_df = metadata[metadata["pert_desc"] == DrOI]
    DrOI_pert_ids = list(DrOI_df["pert_id"])
    DrOI_up_signatures = {k: L1000_up_lookup[k] for k in (DrOI_pert_ids)}
    DrOI_up_no_perts = {k: v for d in DrOI_up_signatures.values() for k, v in d.items()}
    DrOI_up_drug_sigs = list(DrOI_up_no_perts.keys())
    DrOI_down_signatures = {k: L1000_down_lookup[k] for k in (DrOI_pert_ids)}
    DrOI_down_no_perts = {k: v for d in DrOI_down_signatures.values() for k, v in d.items()}
    DrOI_down_drug_sigs = list(DrOI_down_no_perts.keys())
    DrOI_all_sigs = pd.DataFrame(list(set(DrOI_up_drug_sigs) & set (DrOI_down_drug_sigs)))


                
    return render_template('x2kl1000.html', signatures =    DrOI_all_sigs.to_json())

@app.route(ENDPOINT + '/x2k_creeds', methods=['POST'])
def x2k_creeds():
    data = request.get_json(force=True)
    DrOI = data['input']

    CREEDS_URL = 'http://amp.pharm.mssm.edu/CREEDS/'
    CREEEDS_Drug_response = requests.get(CREEDS_URL + 'search', params={'q':DrOI})
    if CREEEDS_Drug_response.status_code == 200:
        CREEDS_drug_output_df = pd.DataFrame(CREEEDS_Drug_response.json())
        CREEDS_drug_output_df['name'] = DrOI
        CREEDS_drug_output_df['full_name'] = CREEDS_drug_output_df[['name', 'geo_id', 'organism']].apply(lambda x: ' '.join(x), axis=1)
        CREEDS_name_df = pd.DataFrame(CREEDS_drug_output_df['full_name'])
        CREEDS_drug_output_ids = pd.DataFrame(CREEDS_drug_output_df['id'])
    #CREEDS_drug_output_df #Display this table
    # CREEDS_drug_sigs = list(CREEDS_drug_output_df.id)
    # DrOI_up_extract = L1000_up_extract[L1000_up_extract['Unnamed: 0'].apply(lambda s: bool(re.compile(DrOI, re.IGNORECASE).search(s)))]
    # DrOI_up_final = DrOI_up_extract.loc[:, (DrOI_up_extract != 0).any(axis=0)] # remove any genes without any expression in any of the results

    # DrOI_down_extract = L1000_down_extract[L1000_down_extract['Unnamed: 0'].apply(lambda s: bool(re.compile(DrOI, re.IGNORECASE).search(s)))]
    # DrOI_down_final = DrOI_down_extract.loc[:, (DrOI_down_extract != 0).any(axis=0)] # remove any genes without any expression in any of the results
    # CREEDS_URL = 'http://amp.pharm.mssm.edu/CREEDS/'
    # CREEDS_Drug_response = requests.get(CREEDS_URL + 'search', params={'q':DrOI})
    # if CREEDS_Drug_response.status_code == 200:
    # #pprint(CREEEDS_Drug_response.json())
    # #json.dump(CREEEDS_Drug_response.json(), open(DrOI + '_api1_result.json', 'w'), indent=4)
    #     CREEDS_drug_output_df = pd.DataFrame(CREEDS_Drug_response.json())
    #     CREEDS_drug_output_df['name'] = DrOI
    #     CREEDS_drug_output_df['full_name'] = CREEDS_drug_output_df[['name', 'geo_id', 'organism']].apply(lambda x: ' '.join(x), axis=1)
    #     CREEDS_name_df = pd.DataFrame(CREEDS_drug_output_df['full_name'])
    #     CREEDS_drug_output_ids = pd.DataFrame(CREEDS_drug_output_df['id'])


    return render_template("x2k_creeds.html", x2k_fullNames = CREEDS_name_df.to_json(), x2k_drugIds = CREEDS_drug_output_ids.to_json())

@app.route(ENDPOINT + '/drug_matrix_drugs', methods=['POST'])
def drugMatrixDrugs():
    data = request.get_json(force=True)
    DrOI = data['input'] 
    DrugMatrix ={}
    for line in urllib.request.urlopen('http://amp.pharm.mssm.edu/lincs-playground/index.php/s/bl6rC50ALsbXfw1/download'):
        label, genelist = line.decode().split('\t\t', maxsplit=1) 
        genelist_split = genelist.strip().split('\t')
        DrugMatrix[label] = genelist_split
    
    DrugMatrix

    ## generate a list of searchable keys to reduce dict
    DrugMatrix_keys = pd.DataFrame(list(DrugMatrix.keys()))
    DrugMatrix_keys.columns = ["sigs"]
    Drug_Matrix_DrOI = DrugMatrix_keys[DrugMatrix_keys["sigs"].apply(lambda s: bool(re.compile(str(DrOI), re.IGNORECASE).search(str(s))))]
    ## reduce dict
    Drug_matrix_sigs_reduced = list(Drug_Matrix_DrOI["sigs"])
    #DrugMatrix_sigs = {k: DrugMatrix[k] for k in list(Drug_matrix_sigs_reduced["sigs"])} # total sigs
    ## up sigs
    Drug_matrix_up_sigs_reduced = Drug_Matrix_DrOI[Drug_Matrix_DrOI["sigs"].apply(lambda s: bool(re.compile(str("-up"), re.IGNORECASE).search(str(s))))]
    DrugMatrix_up_sigs= {k: DrugMatrix[k] for k in list(Drug_matrix_up_sigs_reduced["sigs"])}
    DrugMatrix_sigs = [sig[:-3] for sig in DrugMatrix_up_sigs]
    DrugMatrix_sigs_df = pd.DataFrame(DrugMatrix_sigs)
    DrugMatrix_up_sigs_df = pd.DataFrame(list(Drug_matrix_up_sigs_reduced["sigs"]))


                                        

    for a in list(Drug_matrix_up_sigs_reduced["sigs"]):
        DrugMatrix_up_sigs_save = DrugMatrix_up_sigs[a]
        DrugMatrix_up_df = pd.DataFrame(DrugMatrix_up_sigs_save)
        print(DrugMatrix_up_df)
        #with open(a + "_DrugMatrix_up_sig.json", "w") as f:
            #json.dump(DrugMatrix_up_sigs_save, f)

    ## down sigs
    Drug_matrix_down_sigs_reduced = Drug_Matrix_DrOI[Drug_Matrix_DrOI["sigs"].apply(lambda s: bool(re.compile(str("-dn"), re.IGNORECASE).search(str(s))))]
    DrugMatrix_down_sigs= {k: DrugMatrix[k] for k in list(Drug_matrix_down_sigs_reduced["sigs"])}
    DrugMatrix_down_sigs_df = pd.DataFrame(list(Drug_matrix_down_sigs_reduced["sigs"]))

    
                                    
    for b in list(Drug_matrix_down_sigs_reduced["sigs"]):
        DrugMatrix_down_sigs_save = DrugMatrix_down_sigs[b]
        print(b)
        #with open(b + "_DrugMatrix_down_sig.json", "w") as f:
            #json.dump(DrugMatrix_down_sigs_save, f)
    return render_template('drugMatrix_drugs.html', drOI = DrOI, drugSigs = DrugMatrix_sigs_df.to_json(), drugUpSigs = DrugMatrix_up_sigs_df.to_json(), drugDownSigs = DrugMatrix_down_sigs_df.to_json())

def return_drugMatrix_up_df(sig_id, drOI):
    DrugMatrix ={}
    for line in urllib.request.urlopen('http://amp.pharm.mssm.edu/lincs-playground/index.php/s/bl6rC50ALsbXfw1/download'):
        label, genelist = line.decode().split('\t\t', maxsplit=1) 
        genelist_split = genelist.strip().split('\t')
        DrugMatrix[label] = genelist_split
    
    ## generate a list of searchable keys to reduce dict
    DrugMatrix_keys = pd.DataFrame(list(DrugMatrix.keys()))
    DrugMatrix_keys.columns = ["sigs"]
    Drug_Matrix_DrOI = DrugMatrix_keys[DrugMatrix_keys["sigs"].apply(lambda s: bool(re.compile(str(drOI), re.IGNORECASE).search(str(s))))]
    ## reduce dict
    Drug_matrix_sigs_reduced = list(Drug_Matrix_DrOI["sigs"])
    #DrugMatrix_sigs = {k: DrugMatrix[k] for k in list(Drug_matrix_sigs_reduced["sigs"])} # total sigs
    ## up sigs
    Drug_matrix_up_sigs_reduced = Drug_Matrix_DrOI[Drug_Matrix_DrOI["sigs"].apply(lambda s: bool(re.compile(str("-up"), re.IGNORECASE).search(str(s))))]
    DrugMatrix_up_sigs= {k: DrugMatrix[k] for k in list(Drug_matrix_up_sigs_reduced["sigs"])}
    DrugMatrix_sigs = [sig[:-3] for sig in DrugMatrix_up_sigs]
    DrugMatrix_sigs_df = pd.DataFrame(DrugMatrix_sigs)
    DrugMatrix_up_sigs_df = pd.DataFrame(list(Drug_matrix_up_sigs_reduced["sigs"]))
    DrugMatrix_up_sigs_save = DrugMatrix_up_sigs[str(sig_id)]
    DrugMatrix_up_df = pd.DataFrame(DrugMatrix_up_sigs_save)
    DrugMatrix_up_df.columns = ['Gene']
    return DrugMatrix_up_df

def return_drugMatrix_down_df(sig_id, drOI):
    DrugMatrix ={}
    for line in urllib.request.urlopen('http://amp.pharm.mssm.edu/lincs-playground/index.php/s/bl6rC50ALsbXfw1/download'):
        label, genelist = line.decode().split('\t\t', maxsplit=1) 
        genelist_split = genelist.strip().split('\t')
        DrugMatrix[label] = genelist_split
    
    ## generate a list of searchable keys to reduce dict
    DrugMatrix_keys = pd.DataFrame(list(DrugMatrix.keys()))
    DrugMatrix_keys.columns = ["sigs"]
    Drug_Matrix_DrOI = DrugMatrix_keys[DrugMatrix_keys["sigs"].apply(lambda s: bool(re.compile(str(drOI), re.IGNORECASE).search(str(s))))]
    ## reduce dict
    Drug_matrix_sigs_reduced = list(Drug_Matrix_DrOI["sigs"])
    Drug_matrix_down_sigs_reduced = Drug_Matrix_DrOI[Drug_Matrix_DrOI["sigs"].apply(lambda s: bool(re.compile(str("-dn"), re.IGNORECASE).search(str(s))))]
    DrugMatrix_down_sigs= {k: DrugMatrix[k] for k in list(Drug_matrix_down_sigs_reduced["sigs"])}
    DrugMatrix_down_sigs_df = pd.DataFrame(list(Drug_matrix_down_sigs_reduced["sigs"]))
    DrugMatrix_down_sigs_save = DrugMatrix_down_sigs[str(sig_id)]
    DrugMatrix_down_df = pd.DataFrame(DrugMatrix_down_sigs_save)
    DrugMatrix_down_df.columns = ['Gene']
    return DrugMatrix_down_df


@app.route(ENDPOINT + '/drug_matrix_diseases', methods=['POST'])
def drugMatrixDiseases():
    data = request.get_json(force=True)
    DOI = data['input']

    DrugMatrix ={}
    for line in urllib.request.urlopen('http://amp.pharm.mssm.edu/lincs-playground/index.php/s/bl6rC50ALsbXfw1/download'):
        label, genelist = line.decode().split('\t\t', maxsplit=1) 
        genelist_split = genelist.strip().split('\t')
        DrugMatrix[label] = genelist_split

    DOI_ICD9 = icd9_to_doid_final[icd9_to_doid_final.Disease.apply(lambda s: bool(re.compile(DOI, re.IGNORECASE).search(s)))]
    DOI_ICD9_codes = DOI_ICD9.ICD9
    # get the do_id from the DOI
    DOI_DOID_codes = DOI_ICD9.DOID
    # get the do_id from the DOI
    DOI_CREEDS_codes = DOI_ICD9.CREEDS_drug_id
        
    ##filter by DOI. Need to convert DOI to ICD9 first.
    icd9_to_doid_final_search = icd9_to_doid_final[icd9_to_doid_final["Disease"].apply(lambda s: bool(re.compile(DOI, re.IGNORECASE).search(s)))]
    icd9_to_doid_final_search1 = icd9_to_doid_final_search["ICD9"]
    ## rebuild the wildcard dataframe
    icd9_wildcard = pd.DataFrame(icd9_to_doid_final_search1, columns=['ICD9'])
    icd9_wildcard['ICD9_wildcard'] = icd9_wildcard['ICD9'].apply(lambda code: str(code).split('.')[0])
    icd9_wildcard.head()
    icd9_to_doid_final['ICD9_wildcard'] = icd9_to_doid_final['ICD9'].apply(lambda code: str(code).split('.')[0])
    ICD9_df_joined = pd.merge(
        left=icd9_wildcard, left_on='ICD9_wildcard',
        right=icd9_to_doid_final, right_on='ICD9_wildcard',
        how='inner',
        suffixes=(
            '_Manual',
            '_right',
        )
    )
    emr_sub = EMR_data_df[EMR_data_df["Description"].apply(lambda s: bool(re.compile(str(DOI), re.IGNORECASE).search(str(s))))]
    emr_sub.reset_index(drop = True, inplace = True)
    emr_sub = []
    for a in ICD9_df_joined.ICD9_wildcard.unique():
        emr_sub1 = EMR_data_df[EMR_data_df["ICD9"].apply(lambda s: bool(re.compile(str(a), re.IGNORECASE).search(str(s))))]
        emr_sub.append(emr_sub1)
    emr_sub_df = pd.concat(emr_sub)
    top_drugs_from_disease = list(emr_sub_df.Drug_Name[0:10]) #take the top 5 drugs
    top_drugs_from_disease
    single_word_drugs = []
    for i in top_drugs_from_disease:
        j = str(i)
        splitted = j.split()
        first_word = splitted[0]
        single_word_drugs.append(first_word) 
    single_word_drugs = list(set(single_word_drugs))
    single_word_drugs_list = list(pd.Series(single_word_drugs).map(process_blacklist))
    DrugMatrix_keys = pd.DataFrame(list(DrugMatrix.keys()))
    DrugMatrix_keys.columns = ["sigs"]
    total_drugs = []
    single_word_drugs_list = strip_list_noempty(single_word_drugs_list)
    for q in single_word_drugs_list:
        Drug_Matrix_DrOI = DrugMatrix_keys[DrugMatrix_keys["sigs"].apply(lambda s: bool(re.compile(str(q), re.IGNORECASE).search(str(s))))]
        total_drugs.append((Drug_Matrix_DrOI))
    total_drugs1 = pd.concat(total_drugs)
    ## reduce dict
    Drug_Matrix_DrOI = total_drugs1
    Drug_matrix_sigs_reduced = list(Drug_Matrix_DrOI["sigs"])
    
    #DrugMatrix_sigs = {k: DrugMatrix[k] for k in list(Drug_matrix_sigs_reduced["sigs"])} # total sigs
    ## up sigs
    Drug_matrix_up_sigs_reduced = Drug_Matrix_DrOI[Drug_Matrix_DrOI["sigs"].apply(lambda s: bool(re.compile(str("-up"), re.IGNORECASE).search(str(s))))]
    DrugMatrix_up_sigs= {k: DrugMatrix[k] for k in list(Drug_matrix_up_sigs_reduced["sigs"])}
    DrugMatrix_sigs = [sig[:-3] for sig in DrugMatrix_up_sigs]
    DrugMatrix_sigs_df = pd.DataFrame(DrugMatrix_sigs)
    DrugMatrix_up_sigs_df = pd.DataFrame(list(Drug_matrix_up_sigs_reduced["sigs"]))
    for a in list(Drug_matrix_up_sigs_reduced["sigs"]):
        DrugMatrix_up_sigs_save = DrugMatrix_up_sigs[a]
        print(a)
        #with open(a + "_DrugMatrix_up_sig.json", "w") as f:
            #json.dump(DrugMatrix_up_sigs_save, f)
            
    ## down sigs
    Drug_matrix_down_sigs_reduced = Drug_Matrix_DrOI[Drug_Matrix_DrOI["sigs"].apply(lambda s: bool(re.compile(str("-dn"), re.IGNORECASE).search(str(s))))]
    DrugMatrix_down_sigs= {k: DrugMatrix[k] for k in list(Drug_matrix_down_sigs_reduced["sigs"])}
    DrugMatrix_down_sigs_df = pd.DataFrame(list(Drug_matrix_down_sigs_reduced["sigs"]))
    for b in list(Drug_matrix_down_sigs_reduced["sigs"]):
        DrugMatrix_down_sigs_save = DrugMatrix_down_sigs[b]
        print(b)
        #with open(b + "_DrugMatrix_down_sig.json", "w") as f:
            #json.dump(DrugMatrix_down_sigs_save, f)

    return render_template('drugMatrix_diseases.html', doi = DOI, drugMatrixSigs = DrugMatrix_sigs_df.to_json(), drugMatrixUpSigs = DrugMatrix_up_sigs_df.to_json(), drugMatrixDownSigs = DrugMatrix_down_sigs_df.to_json())

def return_drugMatrix_diseases_up_df(sig_id, doi):
    DrugMatrix ={}
    for line in urllib.request.urlopen('http://amp.pharm.mssm.edu/lincs-playground/index.php/s/bl6rC50ALsbXfw1/download'):
        label, genelist = line.decode().split('\t\t', maxsplit=1) 
        genelist_split = genelist.strip().split('\t')
        DrugMatrix[label] = genelist_split

    DOI_ICD9 = icd9_to_doid_final[icd9_to_doid_final.Disease.apply(lambda s: bool(re.compile(doi, re.IGNORECASE).search(s)))]
    DOI_ICD9_codes = DOI_ICD9.ICD9
    # get the do_id from the DOI
    DOI_DOID_codes = DOI_ICD9.DOID
    # get the do_id from the DOI
    DOI_CREEDS_codes = DOI_ICD9.CREEDS_drug_id
        
    ##filter by DOI. Need to convert DOI to ICD9 first.
    icd9_to_doid_final_search = icd9_to_doid_final[icd9_to_doid_final["Disease"].apply(lambda s: bool(re.compile(doi, re.IGNORECASE).search(s)))]
    icd9_to_doid_final_search1 = icd9_to_doid_final_search["ICD9"]
    ## rebuild the wildcard dataframe
    icd9_wildcard = pd.DataFrame(icd9_to_doid_final_search1, columns=['ICD9'])
    icd9_wildcard['ICD9_wildcard'] = icd9_wildcard['ICD9'].apply(lambda code: str(code).split('.')[0])
    icd9_wildcard.head()
    icd9_to_doid_final['ICD9_wildcard'] = icd9_to_doid_final['ICD9'].apply(lambda code: str(code).split('.')[0])
    ICD9_df_joined = pd.merge(
        left=icd9_wildcard, left_on='ICD9_wildcard',
        right=icd9_to_doid_final, right_on='ICD9_wildcard',
        how='inner',
        suffixes=(
            '_Manual',
            '_right',
        )
    )
    emr_sub = EMR_data_df[EMR_data_df["Description"].apply(lambda s: bool(re.compile(str(doi), re.IGNORECASE).search(str(s))))]
    emr_sub.reset_index(drop = True, inplace = True)
    emr_sub = []
    for a in ICD9_df_joined.ICD9_wildcard.unique():
        emr_sub1 = EMR_data_df[EMR_data_df["ICD9"].apply(lambda s: bool(re.compile(str(a), re.IGNORECASE).search(str(s))))]
        emr_sub.append(emr_sub1)
    emr_sub_df = pd.concat(emr_sub)
    top_drugs_from_disease = list(emr_sub_df.Drug_Name[0:10]) #take the top 5 drugs
    top_drugs_from_disease
    single_word_drugs = []
    for i in top_drugs_from_disease:
        j = str(i)
        splitted = j.split()
        first_word = splitted[0]
        single_word_drugs.append(first_word) 
    single_word_drugs = list(set(single_word_drugs))
    single_word_drugs_list = list(pd.Series(single_word_drugs).map(process_blacklist))
    DrugMatrix_keys = pd.DataFrame(list(DrugMatrix.keys()))
    DrugMatrix_keys.columns = ["sigs"]
    total_drugs = []
    single_word_drugs_list = strip_list_noempty(single_word_drugs_list)
    for q in single_word_drugs_list:
        Drug_Matrix_DrOI = DrugMatrix_keys[DrugMatrix_keys["sigs"].apply(lambda s: bool(re.compile(str(q), re.IGNORECASE).search(str(s))))]
        total_drugs.append((Drug_Matrix_DrOI))
    total_drugs1 = pd.concat(total_drugs)
    ## reduce dict
    Drug_Matrix_DrOI = total_drugs1
    Drug_matrix_sigs_reduced = list(Drug_Matrix_DrOI["sigs"])
    
    #DrugMatrix_sigs = {k: DrugMatrix[k] for k in list(Drug_matrix_sigs_reduced["sigs"])} # total sigs
    ## up sigs
    Drug_matrix_up_sigs_reduced = Drug_Matrix_DrOI[Drug_Matrix_DrOI["sigs"].apply(lambda s: bool(re.compile(str("-up"), re.IGNORECASE).search(str(s))))]
    DrugMatrix_up_sigs= {k: DrugMatrix[k] for k in list(Drug_matrix_up_sigs_reduced["sigs"])}
    DrugMatrix_sigs = [sig[:-3] for sig in DrugMatrix_up_sigs]
    DrugMatrix_sigs_df = pd.DataFrame(DrugMatrix_sigs)
    DrugMatrix_up_sigs_df = pd.DataFrame(list(Drug_matrix_up_sigs_reduced["sigs"]))
    # for a in list(Drug_matrix_up_sigs_reduced["sigs"]):
    DrugMatrix_up_sigs_save = DrugMatrix_up_sigs[str(sig_id)]
    DrugMatrix_up_df = pd.DataFrame(DrugMatrix_up_sigs_save)
    DrugMatrix_up_df.columns = ['Gene']

    return DrugMatrix_up_df

def return_drugMatrix_diseases_down_df(sig_id, doi):
    DrugMatrix ={}
    for line in urllib.request.urlopen('http://amp.pharm.mssm.edu/lincs-playground/index.php/s/bl6rC50ALsbXfw1/download'):
        label, genelist = line.decode().split('\t\t', maxsplit=1) 
        genelist_split = genelist.strip().split('\t')
        DrugMatrix[label] = genelist_split

    DOI_ICD9 = icd9_to_doid_final[icd9_to_doid_final.Disease.apply(lambda s: bool(re.compile(doi, re.IGNORECASE).search(s)))]
    DOI_ICD9_codes = DOI_ICD9.ICD9
    # get the do_id from the DOI
    DOI_DOID_codes = DOI_ICD9.DOID
    # get the do_id from the DOI
    DOI_CREEDS_codes = DOI_ICD9.CREEDS_drug_id
        
    ##filter by DOI. Need to convert DOI to ICD9 first.
    icd9_to_doid_final_search = icd9_to_doid_final[icd9_to_doid_final["Disease"].apply(lambda s: bool(re.compile(doi, re.IGNORECASE).search(s)))]
    icd9_to_doid_final_search1 = icd9_to_doid_final_search["ICD9"]
    ## rebuild the wildcard dataframe
    icd9_wildcard = pd.DataFrame(icd9_to_doid_final_search1, columns=['ICD9'])
    icd9_wildcard['ICD9_wildcard'] = icd9_wildcard['ICD9'].apply(lambda code: str(code).split('.')[0])
    icd9_wildcard.head()
    icd9_to_doid_final['ICD9_wildcard'] = icd9_to_doid_final['ICD9'].apply(lambda code: str(code).split('.')[0])
    ICD9_df_joined = pd.merge(
        left=icd9_wildcard, left_on='ICD9_wildcard',
        right=icd9_to_doid_final, right_on='ICD9_wildcard',
        how='inner',
        suffixes=(
            '_Manual',
            '_right',
        )
    )
    emr_sub = EMR_data_df[EMR_data_df["Description"].apply(lambda s: bool(re.compile(str(doi), re.IGNORECASE).search(str(s))))]
    emr_sub.reset_index(drop = True, inplace = True)
    emr_sub = []
    for a in ICD9_df_joined.ICD9_wildcard.unique():
        emr_sub1 = EMR_data_df[EMR_data_df["ICD9"].apply(lambda s: bool(re.compile(str(a), re.IGNORECASE).search(str(s))))]
        emr_sub.append(emr_sub1)
    emr_sub_df = pd.concat(emr_sub)
    top_drugs_from_disease = list(emr_sub_df.Drug_Name[0:10]) #take the top 5 drugs
    top_drugs_from_disease
    single_word_drugs = []
    for i in top_drugs_from_disease:
        j = str(i)
        splitted = j.split()
        first_word = splitted[0]
        single_word_drugs.append(first_word) 
    single_word_drugs = list(set(single_word_drugs))
    single_word_drugs_list = list(pd.Series(single_word_drugs).map(process_blacklist))
    DrugMatrix_keys = pd.DataFrame(list(DrugMatrix.keys()))
    DrugMatrix_keys.columns = ["sigs"]
    total_drugs = []
    single_word_drugs_list = strip_list_noempty(single_word_drugs_list)
    for q in single_word_drugs_list:
        Drug_Matrix_DrOI = DrugMatrix_keys[DrugMatrix_keys["sigs"].apply(lambda s: bool(re.compile(str(q), re.IGNORECASE).search(str(s))))]
        total_drugs.append((Drug_Matrix_DrOI))
    total_drugs1 = pd.concat(total_drugs)
    ## reduce dict
    Drug_Matrix_DrOI = total_drugs1
    Drug_matrix_sigs_reduced = list(Drug_Matrix_DrOI["sigs"])
    
    #DrugMatrix_sigs = {k: DrugMatrix[k] for k in list(Drug_matrix_sigs_reduced["sigs"])} # total sigs
    ## up sigs
    
    ## down sigs
    Drug_matrix_down_sigs_reduced = Drug_Matrix_DrOI[Drug_Matrix_DrOI["sigs"].apply(lambda s: bool(re.compile(str("-dn"), re.IGNORECASE).search(str(s))))]
    DrugMatrix_down_sigs= {k: DrugMatrix[k] for k in list(Drug_matrix_down_sigs_reduced["sigs"])}
    DrugMatrix_down_sigs_df = pd.DataFrame(list(Drug_matrix_down_sigs_reduced["sigs"]))
    DrugMatrix_down_sigs_save = DrugMatrix_down_sigs[str(sig_id)]
    DrugMatrix_down_df = pd.DataFrame(DrugMatrix_down_sigs_save)
    DrugMatrix_down_df.columns = ['Gene']    
        
    return DrugMatrix_down_df

def strip_list_noempty(mylist):
    newlist = (item.strip() if hasattr(item, 'strip') else item for item in mylist)
    return [item for item in newlist if item != '']

@app.route(ENDPOINT + '/x2k_L1000_diseases', methods= ['POST'])
def x2k_L1000_diseases():
    data = request.get_json(force=True)
    DOI = data['input'] # disease of interest. CAN TAKE FROM posssible_disease_list FOR NOW
####

    ### DISEASE --> ICD9 --> DOI
    # get the ICD9 from the DOI
    DOI_ICD9 = icd9_to_doid_final[icd9_to_doid_final.Disease.apply(lambda s: bool(re.compile(DOI, re.IGNORECASE).search(s)))]
    DOI_ICD9_codes = DOI_ICD9.ICD9
    # get the do_id from the DOI
    DOI_DOID_codes = DOI_ICD9.DOID
    # get the do_id from the DOI
    DOI_CREEDS_codes = DOI_ICD9.CREEDS_drug_id
        
    ##filter by DOI. Need to convert DOI to ICD9 first.
    icd9_to_doid_final_search = icd9_to_doid_final[icd9_to_doid_final["Disease"].apply(lambda s: bool(re.compile(DOI, re.IGNORECASE).search(s)))]
    icd9_to_doid_final_search1 = icd9_to_doid_final_search["ICD9"]
    ## rebuild the wildcard dataframe
    icd9_wildcard = pd.DataFrame(icd9_to_doid_final_search1, columns=['ICD9'])
    icd9_wildcard['ICD9_wildcard'] = icd9_wildcard['ICD9'].apply(lambda code: str(code).split('.')[0])
    icd9_wildcard.head()
    icd9_to_doid_final['ICD9_wildcard'] = icd9_to_doid_final['ICD9'].apply(lambda code: str(code).split('.')[0])
    ICD9_df_joined = pd.merge(
        left=icd9_wildcard, left_on='ICD9_wildcard',
        right=icd9_to_doid_final, right_on='ICD9_wildcard',
        how='inner',
        suffixes=(
            '_Manual',
            '_right',
        )
    )
    emr_sub = EMR_data_df[EMR_data_df["Description"].apply(lambda s: bool(re.compile(str(DOI), re.IGNORECASE).search(str(s))))]
    emr_sub.reset_index(drop = True, inplace = True)
    emr_sub = []
    for a in ICD9_df_joined.ICD9_wildcard.unique():
        emr_sub1 = EMR_data_df[EMR_data_df["ICD9"].apply(lambda s: bool(re.compile(str(a), re.IGNORECASE).search(str(s))))]
        emr_sub.append(emr_sub1)
    emr_sub_df = pd.concat(emr_sub)
    top_drugs_from_disease = list(emr_sub_df.Drug_Name[0:15]) #take the top 5 drugs
    top_drugs_from_disease
    single_word_drugs = []
    for i in top_drugs_from_disease:
        j = str(i)
        splitted = j.split()
        first_word = splitted[0]
        single_word_drugs.append(first_word) 
    single_word_drugs = list(set(single_word_drugs))
    single_word_drugs_list = list(pd.Series(single_word_drugs).map(process_blacklist))
    


    L1000_reverse_drugs_store = []
    L1000FWD_URL = 'http://amp.pharm.mssm.edu/L1000FWD/'
    L1000_reverse_drugs_store = []
    L1000_reverse_pertids_store = []
    drug_and_perts = pd.DataFrame()
    for a in single_word_drugs_list:
        query_string = a
        L1000_reverse_drug_response = requests.get(L1000FWD_URL + 'synonyms/' + query_string)
        if L1000_reverse_drug_response.status_code == 200:
            #pprint(L1000_reverse_drug_response.json())
            L1000_reverse_significant_query = L1000_reverse_drug_response.json()
            if len(L1000_reverse_significant_query) > 0:
                #json.dump(L1000_reverse_drug_response.json(), open(query_string + '_L1000_reverse_drug_query.json', 'w'), indent=4)
                L1000_reverse_significant_query = L1000_reverse_drug_response.json()
                L1000_reverse_significant_query_df = pd.DataFrame(L1000_reverse_significant_query)
                L1000_reverse_pertids_store.append(list(L1000_reverse_significant_query_df["pert_id"]))
                
                # store the drug name with the pert ID
                drug_and_perts = drug_and_perts.append(L1000_reverse_significant_query_df)
                
                L1000_reverse_drugs_store.append(a)            
                print("Found significant L1000 drug signatures for " + query_string)            
            else:
                print("No significant L1000 drug signatures for " + query_string)     
    
        
    L1000_reverse_pertids_flat = []
    for sublist in L1000_reverse_pertids_store:
        for item in sublist:
            L1000_reverse_pertids_flat.append(item) 
    L1000_reverse_pertids_flat = set(list(L1000_reverse_pertids_flat)) & set(list(metadata["pert_id"])) & set(list(L1000_up_lookup.keys()))
    DrOI_disease_up_signatures = {k: L1000_up_lookup[k] for k in (L1000_reverse_pertids_flat)}
    DrOI_disease_up_no_perts = {k: v for d in DrOI_disease_up_signatures.values() for k, v in d.items()}
    DrOI_disease_up_drug_sigs = list(DrOI_disease_up_no_perts.keys())
    DrOI_disease_down_signatures = {k: L1000_down_lookup[k] for k in (L1000_reverse_pertids_flat)}
    DrOI_disease_down_no_perts = {k: v for d in DrOI_disease_down_signatures.values() for k, v in d.items()}
    DrOI_disease_down_drug_sigs = list(DrOI_disease_down_no_perts.keys())
    DrOI_disease_all_sigs = set(DrOI_disease_up_drug_sigs) & set(DrOI_disease_down_drug_sigs)
    L1000_disease_signatures = pd.DataFrame(list(DrOI_disease_all_sigs))

    return render_template('x2k_L1000_diseases.html', x2k_L1000_diseases_sig_ids = L1000_disease_signatures.to_json(), doi = DOI)

@app.route(ENDPOINT + '/x2k_creeds_rx_diseases', methods=["POST"])
def x2k_creeds_rx_diseases():
    data = request.get_json(force=True)
    DOI = data['input']
    ### DISEASE --> ICD9 --> DOI
    # get the ICD9 from the DOI
    DOI_ICD9 = icd9_to_doid_final[icd9_to_doid_final.Disease.apply(lambda s: bool(re.compile(DOI, re.IGNORECASE).search(s)))]
    DOI_ICD9_codes = DOI_ICD9.ICD9
    # get the do_id from the DOI
    DOI_DOID_codes = DOI_ICD9.DOID
    # get the do_id from the DOI
    DOI_CREEDS_codes = DOI_ICD9.CREEDS_drug_id
        
    ##filter by DOI. Need to convert DOI to ICD9 first.
    icd9_to_doid_final_search = icd9_to_doid_final[icd9_to_doid_final["Disease"].apply(lambda s: bool(re.compile(DOI, re.IGNORECASE).search(s)))]
    icd9_to_doid_final_search1 = icd9_to_doid_final_search["ICD9"]
    ## rebuild the wildcard dataframe
    icd9_wildcard = pd.DataFrame(icd9_to_doid_final_search1, columns=['ICD9'])
    icd9_wildcard['ICD9_wildcard'] = icd9_wildcard['ICD9'].apply(lambda code: str(code).split('.')[0])
    icd9_wildcard.head()
    icd9_to_doid_final['ICD9_wildcard'] = icd9_to_doid_final['ICD9'].apply(lambda code: str(code).split('.')[0])
    #icd9_to_doid_final.head()
    ICD9_df_joined = pd.merge(
        left=icd9_wildcard, left_on='ICD9_wildcard',
        right=icd9_to_doid_final, right_on='ICD9_wildcard',
        how='inner',
        suffixes=(
            '_Manual',
            '_right',
        )
    )
    #ICD9_codes = str(int(ICD9_df_joined["ICD9_wildcard"].unique())) 
    ## generate an emr based on the ICD_9 codes extracted; can now extract the drug names as well
    #emr_sub = EMR_data_df[EMR_data_df['ICD9'].apply(lambda s: bool(re.compile(ICD9_codes, re.IGNORECASE).search(s)))]
    emr_sub = EMR_data_df[EMR_data_df["Description"].apply(lambda s: bool(re.compile(str(DOI), re.IGNORECASE).search(str(s))))]
    #emr_sub[0:10]
    emr_sub.reset_index(drop = True, inplace = True)
    emr_sub = []
    for a in ICD9_df_joined.ICD9_wildcard.unique():
        emr_sub1 = EMR_data_df[EMR_data_df["ICD9"].apply(lambda s: bool(re.compile(str(a), re.IGNORECASE).search(str(s))))]
        emr_sub.append(emr_sub1)
    emr_sub_df = pd.concat(emr_sub)
    #### L1000 integration
    # disease to drug conversion (disease input)
    top_drugs_from_disease = list(emr_sub_df.Drug_Name[0:10]) #take the top 5 drugs
    
## same as LINC1000h5.row_metadata_df
    #metadata
    for a in top_drugs_from_disease:
        print(a)
        meta_doi = metadata[metadata["pert_desc"].apply(lambda s: bool(re.compile(str(a), re.IGNORECASE).search(str(s))))]
    meta_doi
    meta_doi_ids = meta_doi.rid
    query = list(meta_doi_ids)
    #print(query)
    # disease to drug conversion (disease input)
    top_drugs_from_disease = list(emr_sub_df.Drug_Name[0:20]) #take the top 5 drugs
    top_drugs_from_disease
    single_word_drugs = []
    for i in top_drugs_from_disease:
        j = str(i)
        splitted = j.split()
        first_word = splitted[0]
    # print(first_word)
        single_word_drugs.append(first_word) 
    #print(single_word_drugs)
    single_word_drugs = list(set(single_word_drugs))
    single_word_drugs_list = list(pd.Series(single_word_drugs).map(process_blacklist))
    #single_word_drugs

    ### CREEDS DRUG CARD 
    while("" in single_word_drugs_list):
        single_word_drugs_list.remove("")
        
    single_word_drugs_list
    CREEDS_URL = 'http://amp.pharm.mssm.edu/CREEDS/'
    CREEDS_drug_from_disease_up_genes = []
    CREEDS_drug_from_disease_down_genes = []
    up_signature_names = []
    down_signature_names = []
    CREEDS_drug_ids_store = []
    for drug_name in single_word_drugs_list:
        CREEEDS_Drug_response = requests.get(CREEDS_URL + 'search', params={'q':str(drug_name)})
        if CREEEDS_Drug_response.status_code == 200:
            CREEDS_drug_output_df = pd.DataFrame(CREEEDS_Drug_response.json())
            
            if len(CREEDS_drug_output_df) > 0:
                CREEDS_drug_output_ids = list(CREEDS_drug_output_df["id"])
                CREEDS_drug_output_ids_df = pd.DataFrame(CREEDS_drug_output_ids)
                CREEDS_drug_ids_store.append(CREEDS_drug_output_ids)
                CREEDS_drug_output_ids_df = pd.DataFrame(list(itertools.chain(*CREEDS_drug_ids_store)))

                print("CREEDS IDs found for " + drug_name)
                for outputid in CREEDS_drug_output_ids:
                    CREEDS_drug_sigs_response = requests.get(CREEDS_URL + 'api', params={'id':outputid})
                    if CREEDS_drug_sigs_response.status_code == 200:
                        CREEDS_drug_sigs_response_json = CREEDS_drug_sigs_response.json()
                        ## up genes
                        CREEDS_drug_sigs_up_genes = CREEDS_drug_sigs_response_json['up_genes']
                        CREEDS_drug_sigs_up_genes_df = pd.DataFrame(CREEDS_drug_sigs_up_genes) # this is the up genes dataframe
                        CREEDS_drug_sigs_up_genes_df.columns = ["Genes", "Score"]
                        CREEDS_query_up_genes = list(CREEDS_drug_sigs_up_genes_df["Genes"])
                        
                        filename1 = (drug_name + "_CREEDS_drug_sig_up_genes.csv")
                        up_signature_names_temp = (DOI + "_" + 
                                                drug_name + "_" + 
                                                CREEDS_drug_sigs_response_json["geo_id"] + "_" +
                                                CREEDS_drug_sigs_response_json["id"]
                                                )
                        up_signature_names.append(up_signature_names_temp)
                        up_signatures_df = pd.DataFrame(up_signature_names)
                        #CREEDS_drug_sigs_up_genes_df.to_csv(filename1) # this saves the df as a csv
                        ## down genes
                        CREEDS_drug_sigs_down_genes = CREEDS_drug_sigs_response_json['down_genes']
                        CREEDS_drug_sigs_down_genes_df = pd.DataFrame(CREEDS_drug_sigs_down_genes)# this is the down genes dataframe
                        CREEDS_drug_sigs_down_genes_df.columns = ["Genes", "Score"]
                        CREEDS_query_down_genes = list(CREEDS_drug_sigs_down_genes_df["Genes"])
                        
                        filename2 = (drug_name + "_CREEDS_drug_sig_down_genes.csv")
                        down_signature_names_temp = (DOI + "_" + 
                                                drug_name + "_" +
                                                CREEDS_drug_sigs_response_json["geo_id"]+ "_" +
                                                CREEDS_drug_sigs_response_json["id"]
                                                )
                        #print(down_signature_names_temp)
                        down_signature_names.append(down_signature_names_temp)
                        down_signatures_df = pd.DataFrame(down_signature_names)
                        
                        
                        
                        ## need to clean the creeds genes because they have p values attached
                        ## X2K implementation
                        ## up genes
                        CREEDS_X2K_up_genes = run_X2K(CREEDS_query_up_genes)
                        CREEDS_X2K_up_genes = CREEDS_X2K_up_genes["X2K"]
                        CREEDS_X2K_up_genes_df = pd.DataFrame(CREEDS_X2K_up_genes['nodes'])
                        filename_up = (a + "_CREEDS_X2K_up_genes.csv")
                        #CREEDS_X2K_up_genes_df.to_csv(filename_up) # THIS IS THE FILE THEY SHOULD BE ABLE TO DOWNLOAD
                        print(filename_up)
                        ## down genes
                        CREEDS_X2K_down_genes = run_X2K(CREEDS_query_down_genes)
                        CREEDS_X2K_down_genes = CREEDS_X2K_down_genes["X2K"]
                        CREEDS_X2K_down_genes_df = pd.DataFrame(CREEDS_X2K_down_genes['nodes'])
                        filename_down = (a + "_CREEDS_X2K_down_genes.csv")
                        #CREEDS_X2K_down_genes_df.to_csv(filename_down) # THIS IS THE FILE THEY SHOULD BE ABLE TO DOWNLOAD
            
                        
                        
                        #CREEDS_drug_sigs_down_genes_df.to_csv(filename2)
                        #CREEDS_drug_sigs_down_genes = CREEDS_drug_sigs_response_json['down_genes'] # this saves the df as a csv
                        ## json propagation
                        #pprint(response.json())
                        #json.dump(response.json(), open(a + '_CREEDS_Drug_sig.json', 'w'), indent=4) # if the user wants the entire json, they can download this
                        CREEDS_drug_from_disease_up_genes.append(CREEDS_drug_sigs_up_genes)
                        CREEDS_drug_from_disease_down_genes.append(CREEDS_drug_sigs_down_genes_df)
            else:
                print ("No CREEDS IDs found for " + drug_name)
    return render_template("x2k_creeds_rx_diseases.html", x2kCreedsRxSigs = CREEDS_drug_output_ids_df.to_json(), x2kUpSigs = up_signatures_df.to_json(), x2kDownSigs = down_signatures_df.to_json())

def return_x2k_creeds_rx_diseases_up_df(sig_id):
    CREEDS_URL = 'http://amp.pharm.mssm.edu/CREEDS/'
    CREEDS_drug_sigs_response = requests.get(CREEDS_URL + 'api', params={'id':str(sig_id)})
    if CREEDS_drug_sigs_response.status_code == 200:
        CREEDS_drug_sigs_response_json = CREEDS_drug_sigs_response.json()
        ## up genes
        CREEDS_drug_sigs_up_genes = CREEDS_drug_sigs_response_json['up_genes']
        CREEDS_drug_sigs_up_genes_df = pd.DataFrame(CREEDS_drug_sigs_up_genes) # this is the up genes dataframe
        CREEDS_drug_sigs_up_genes_df.columns = ["Genes", "Score"]
        CREEDS_query_up_genes = list(CREEDS_drug_sigs_up_genes_df["Genes"])
        CREEDS_X2K_up_genes = run_X2K(CREEDS_query_up_genes)
        CREEDS_X2K_up_genes = CREEDS_X2K_up_genes["X2K"]
        CREEDS_X2K_up_genes_df = pd.DataFrame(CREEDS_X2K_up_genes['nodes'])
    return CREEDS_X2K_up_genes_df

def return_x2k_creeds_rx_diseases_down_df(sig_id):
    CREEDS_URL = 'http://amp.pharm.mssm.edu/CREEDS/'
    CREEDS_drug_sigs_response = requests.get(CREEDS_URL + 'api', params={'id':str(sig_id)})
    if CREEDS_drug_sigs_response.status_code == 200:
        CREEDS_drug_sigs_response_json = CREEDS_drug_sigs_response.json()

        #CREEDS_drug_sigs_up_genes_df.to_csv(filename1) # this saves the df as a csv
        ## down genes
        CREEDS_drug_sigs_down_genes = CREEDS_drug_sigs_response_json['down_genes']
        CREEDS_drug_sigs_down_genes_df = pd.DataFrame(CREEDS_drug_sigs_down_genes)# this is the down genes dataframe
        CREEDS_drug_sigs_down_genes_df.columns = ["Genes", "Score"]
        CREEDS_query_down_genes = list(CREEDS_drug_sigs_down_genes_df["Genes"])
        
        ## need to clean the creeds genes because they have p values attached
        ## X2K implementation

        ## down genes
        CREEDS_X2K_down_genes = run_X2K(CREEDS_query_down_genes)
        CREEDS_X2K_down_genes = CREEDS_X2K_down_genes["X2K"]
        CREEDS_X2K_down_genes_df = pd.DataFrame(CREEDS_X2K_down_genes['nodes'])
    return CREEDS_X2K_down_genes_df

@app.route(ENDPOINT + "/x2k_creeds_dx_drugs", methods=["POST"])
def x2k_creeds_dx_drugs():
    data = request.get_json(force=True)
    DrOI = data['input']
    CREEDS_GSE = {
        row['id']: [row['geo_id'], row["disease_name"]]
        for row in CREEDS_data
    }

    droi_search =EMR_data_df[EMR_data_df['Drug_Name'].apply(lambda s: bool(re.compile(DrOI, re.IGNORECASE).search(s)))]
    droi_search_top5 = droi_search[0:5]
    EMR_top_disease_from_drug = droi_search_top5["ICD9"]
    #top_disease_from_drug = EMR_top_disease_from_drug[0:5]
    ## build a datatable of all the ICD-9 CM diagnosis codes families (i.e no decimal points)
    EMR_top_disease_from_drug_df = pd.DataFrame(EMR_top_disease_from_drug, columns=['ICD9'])
    EMR_top_disease_from_drug_df['ICD9_wildcard'] = EMR_top_disease_from_drug_df['ICD9'].apply(lambda code: code.split('.')[0])
    #EMR_top_disease_from_drug_df.head()
    icd9_to_doid_final['ICD9_wildcard'] = icd9_to_doid_final['ICD9'].apply(lambda code: str(code).split('.')[0])
    #icd9_to_doid_final.head()
    df_joined = pd.merge(
        left=EMR_top_disease_from_drug_df, left_on='ICD9_wildcard',
        right=icd9_to_doid_final, right_on='ICD9_wildcard',
        how='inner',
        suffixes=(
            '_left',
            '_right',
        )
    )
    CREEDS_drug_ids = pd.DataFrame(set(df_joined.CREEDS_drug_id))
    CREEDS_drug_ids_list = list(set(df_joined.CREEDS_drug_id))
    CREEDS_drug_ids_df = pd.DataFrame(CREEDS_drug_ids_list)
    CREEDS_Drug_Final = dict((k, CREEDS_GSE[k]) for k in CREEDS_drug_ids_list)
    CREEDS_drug_final_df = pd.DataFrame(CREEDS_Drug_Final).T
    CREEDS_drug_final_df.columns = ["GSE_ID", "DISEASE"]
    #CREEDS_drug_final_df # DISPLAY THIS DATAFRAME
    ### CREEDS DISEASE CARD FROM DRUG INPUT API
    CREEDS_drug_final_diseases = CREEDS_drug_final_df.DISEASE
    CREEDS_drug_final_GSE_ID = CREEDS_drug_final_df.GSE_ID
    ## CREEDS DISEASE CARD FROM DISEASE QUERY 
    loop_iteration = np.arange(0, len(CREEDS_drug_final_diseases))
    loop_iteration = list(loop_iteration)
    CREEDS_total_api_df = []
    x2k_creeds_names = []
    for a in loop_iteration:
        CREEDS_URL = 'http://amp.pharm.mssm.edu/CREEDS/'
        CREEEDS_Disease_response = requests.get(CREEDS_URL + 'api', params={'id':CREEDS_drug_ids_list[a]})
        if CREEEDS_Disease_response.status_code == 200:
            CREEEDS_Disease_response_json = CREEEDS_Disease_response.json()
            
            ## up genes
            CREEDS_disease_sigs_up_genes = CREEEDS_Disease_response_json['up_genes']
            CREEDS_disease_sigs_up_genes_df = pd.DataFrame(CREEDS_disease_sigs_up_genes) # this is the up genes dataframe
            CREEDS_disease_sigs_up_genes_df.columns = ["Genes", "Score"]
            CREEDS_disease_sig_up_genes_list = CREEDS_disease_sigs_up_genes_df["Genes"]
            filename1 = (DrOI + "_" + CREEEDS_Disease_response_json["disease_name"] + "_" +
                str(CREEDS_drug_ids_list[a]) + "_CREEDS_up_genes.csv")
            #CREEDS_disease_sigs_up_genes_df.to_csv(filename1) # this saves the df as a csv
            #print(filename1)
            x2k_creeds_name = DrOI + "_" + CREEEDS_Disease_response_json["disease_name"] + "_" + str(CREEDS_drug_ids_list[a])
            x2k_creeds_names.append(x2k_creeds_name)
            x2k_creeds_names_df = pd.DataFrame(x2k_creeds_names)
            
            ## down genes
            CREEDS_disease_sigs_down_genes = CREEEDS_Disease_response_json['down_genes']
            CREEDS_disease_sigs_down_genes_df = pd.DataFrame(CREEDS_disease_sigs_down_genes) # this is the up genes dataframe
            CREEDS_disease_sigs_down_genes_df.columns = ["Genes", "Score"]
            CREEDS_disease_sig_down_genes_list = CREEDS_disease_sigs_down_genes_df["Genes"]
            filename2 = (DrOI + "_" + CREEEDS_Disease_response_json["disease_name"] + "_" +
                str(CREEDS_drug_ids_list[a]) + "_CREEDS_down_genes.csv")
            #CREEDS_disease_sigs_down_genes_df.to_csv(filename2) # this saves the df as a csv
            
            
            ### X2K
            ## up genes
            CREEDS_X2K_up_genes = run_X2K(CREEDS_disease_sig_up_genes_list)
            CREEDS_X2K_up_genes = CREEDS_X2K_up_genes["X2K"]
            CREEDS_X2K_up_genes_df = pd.DataFrame(CREEDS_X2K_up_genes['nodes'])
            filename_up = (DrOI + "_" + CREEEDS_Disease_response_json["disease_name"] + "_" +
                str(CREEDS_drug_ids_list[a]) +  "_CREEDS_X2K_up_genes.csv")
            #CREEDS_X2K_up_genes_df.to_csv(filename_up) # THIS IS THE FILE THEY SHOULD BE ABLE TO DOWNLOAD
            ## down genes
            CREEDS_X2K_down_genes = run_X2K(CREEDS_disease_sig_down_genes_list)
            CREEDS_X2K_down_genes = CREEDS_X2K_down_genes["X2K"]
            CREEDS_X2K_down_genes_df = pd.DataFrame(CREEDS_X2K_down_genes['nodes'])
            filename_down = (DrOI + "_" + CREEEDS_Disease_response_json["disease_name"] + "_" +
                str(CREEDS_drug_ids_list[a]) +  "_CREEDS_X2K_down_genes.csv")
            #CREEDS_X2K_down_genes_df.to_csv(filename_down) # THIS IS THE FILE THEY SHOULD BE ABLE TO DOWNLOAD
    return render_template('x2k_creeds_dx_drugs.html', x2kCreedsDrugSigs = CREEDS_drug_ids_df.to_json(), x2kCreedsNames = x2k_creeds_names_df.to_json())

def return_x2k_creeds_dx_drugs_up_df(sig_id):
    CREEDS_URL = 'http://amp.pharm.mssm.edu/CREEDS/'
    CREEEDS_Disease_response = requests.get(CREEDS_URL + 'api', params={'id':str(sig_id)})
    if CREEEDS_Disease_response.status_code == 200:
        CREEEDS_Disease_response_json = CREEEDS_Disease_response.json()
        
        ## up genes
        CREEDS_disease_sigs_up_genes = CREEEDS_Disease_response_json['up_genes']
        CREEDS_disease_sigs_up_genes_df = pd.DataFrame(CREEDS_disease_sigs_up_genes) # this is the up genes dataframe
        CREEDS_disease_sigs_up_genes_df.columns = ["Genes", "Score"]
        CREEDS_disease_sig_up_genes_list = CREEDS_disease_sigs_up_genes_df["Genes"]
        #filename1 = (DrOI + "_" + CREEEDS_Disease_response_json["disease_name"] + "_" +
            #str(CREEDS_drug_ids_list[a]) + "_CREEDS_up_genes.csv")
        #CREEDS_disease_sigs_up_genes_df.to_csv(filename1) # this saves the df as a csv
        #print(filename1)
        x2k_creeds_names.append(x2k_creeds_name)
        x2k_creeds_names_df = pd.DataFrame(x2k_creeds_names)
        
        ## down genes
        CREEDS_disease_sigs_down_genes = CREEEDS_Disease_response_json['down_genes']
        CREEDS_disease_sigs_down_genes_df = pd.DataFrame(CREEDS_disease_sigs_down_genes) # this is the up genes dataframe
        CREEDS_disease_sigs_down_genes_df.columns = ["Genes", "Score"]
        CREEDS_disease_sig_down_genes_list = CREEDS_disease_sigs_down_genes_df["Genes"]
        #filename2 = (DrOI + "_" + CREEEDS_Disease_response_json["disease_name"] + "_" + str(CREEDS_drug_ids_list[a]) + "_CREEDS_down_genes.csv")
        #CREEDS_disease_sigs_down_genes_df.to_csv(filename2) # this saves the df as a csv
        
        
        ### X2K
        ## up genes
        CREEDS_X2K_up_genes = run_X2K(CREEDS_disease_sig_up_genes_list)
        CREEDS_X2K_up_genes = CREEDS_X2K_up_genes["X2K"]
        CREEDS_X2K_up_genes_df = pd.DataFrame(CREEDS_X2K_up_genes['nodes'])
        #filename_up = (DrOI + "_" + CREEEDS_Disease_response_json["disease_name"] + "_" +
           # str(CREEDS_drug_ids_list[a]) +  "_CREEDS_X2K_up_genes.csv")
        #CREEDS_X2K_up_genes_df.to_csv(filename_up) # THIS IS THE FILE THEY SHOULD BE ABLE TO DOWNLOAD
        ## down genes
        CREEDS_X2K_down_genes = run_X2K(CREEDS_disease_sig_down_genes_list)
        CREEDS_X2K_down_genes = CREEDS_X2K_down_genes["X2K"]
        CREEDS_X2K_down_genes_df = pd.DataFrame(CREEDS_X2K_down_genes['nodes'])
        #filename_down = (DrOI + "_" + CREEEDS_Disease_response_json["disease_name"] + "_" +
            #str(CREEDS_drug_ids_list[a]) +  "_CREEDS_X2K_down_genes.csv")
        #CREEDS_X2K_down_genes_df.to_csv(filename_down) # THIS IS THE FILE THEY SHOULD BE ABLE TO DOWNLOAD
    return CREEDS_X2K_up_genes_df


def return_x2k_creeds_dx_drugs_down_df(sig_id):
    CREEDS_URL = 'http://amp.pharm.mssm.edu/CREEDS/'
    CREEEDS_Disease_response = requests.get(CREEDS_URL + 'api', params={'id':str(sig_id)})
    if CREEEDS_Disease_response.status_code == 200:
        CREEEDS_Disease_response_json = CREEEDS_Disease_response.json()
        
        ## up genes
        CREEDS_disease_sigs_up_genes = CREEEDS_Disease_response_json['up_genes']
        CREEDS_disease_sigs_up_genes_df = pd.DataFrame(CREEDS_disease_sigs_up_genes) # this is the up genes dataframe
        CREEDS_disease_sigs_up_genes_df.columns = ["Genes", "Score"]
        CREEDS_disease_sig_up_genes_list = CREEDS_disease_sigs_up_genes_df["Genes"]
        #filename1 = (DrOI + "_" + CREEEDS_Disease_response_json["disease_name"] + "_" +
            #str(CREEDS_drug_ids_list[a]) + "_CREEDS_up_genes.csv")
        #CREEDS_disease_sigs_up_genes_df.to_csv(filename1) # this saves the df as a csv
        #print(filename1)
        x2k_creeds_names.append(x2k_creeds_name)
        x2k_creeds_names_df = pd.DataFrame(x2k_creeds_names)
        
        ## down genes
        CREEDS_disease_sigs_down_genes = CREEEDS_Disease_response_json['down_genes']
        CREEDS_disease_sigs_down_genes_df = pd.DataFrame(CREEDS_disease_sigs_down_genes) # this is the up genes dataframe
        CREEDS_disease_sigs_down_genes_df.columns = ["Genes", "Score"]
        CREEDS_disease_sig_down_genes_list = CREEDS_disease_sigs_down_genes_df["Genes"]
        #filename2 = (DrOI + "_" + CREEEDS_Disease_response_json["disease_name"] + "_" + str(CREEDS_drug_ids_list[a]) + "_CREEDS_down_genes.csv")
        #CREEDS_disease_sigs_down_genes_df.to_csv(filename2) # this saves the df as a csv
        
        
        ### X2K
        ## up genes
        CREEDS_X2K_up_genes = run_X2K(CREEDS_disease_sig_up_genes_list)
        CREEDS_X2K_up_genes = CREEDS_X2K_up_genes["X2K"]
        CREEDS_X2K_up_genes_df = pd.DataFrame(CREEDS_X2K_up_genes['nodes'])
        #filename_up = (DrOI + "_" + CREEEDS_Disease_response_json["disease_name"] + "_" +
           # str(CREEDS_drug_ids_list[a]) +  "_CREEDS_X2K_up_genes.csv")
        #CREEDS_X2K_up_genes_df.to_csv(filename_up) # THIS IS THE FILE THEY SHOULD BE ABLE TO DOWNLOAD
        ## down genes
        CREEDS_X2K_down_genes = run_X2K(CREEDS_disease_sig_down_genes_list)
        CREEDS_X2K_down_genes = CREEDS_X2K_down_genes["X2K"]
        CREEDS_X2K_down_genes_df = pd.DataFrame(CREEDS_X2K_down_genes['nodes'])
        #filename_down = (DrOI + "_" + CREEEDS_Disease_response_json["disease_name"] + "_" +
           # str(CREEDS_drug_ids_list[a]) +  "_CREEDS_X2K_down_genes.csv")
        #CREEDS_X2K_down_genes_df.to_csv(filename_down) # THIS IS THE FILE THEY SHOULD BE ABLE TO DOWNLOAD
    return CREEDS_X2K_down_genes_df

@app.route(ENDPOINT + '/x2k_creeds_dx_diseases', methods=["POST"])
def x2k_creeds_dx_diseases():
    data = request.get_json(force=True)
    DOI = data['input'] # disease of interest. CAN TAKE FROM posssible_disease_list FOR NOW
    CREEDS_disease = CREEDS_df[CREEDS_df["disease_name"] == DOI]
    CREEDS_disease_ids = list(CREEDS_disease["id"]) # all the signatures to print out
    CREEDS_disease_ids_df = pd.DataFrame(CREEDS_disease_ids)
    CREEDS_disease_ids_df.columns = ["id"]
    x2kCreedsDxDiseaseNames = []
    for a in CREEDS_disease_ids:
        CREEDS_up_gene_sigs = pd.DataFrame(CREEDS_up_genes[a])
        CREEDS_up_gene_sigs.columns = ["Genes", "Score"]
        CREEDS_disease_sig_up_genes_list = CREEDS_up_gene_sigs["Genes"]
        filename1 = (DOI + "_" + str(a) + "_CREEDS_disease_sig_up_genes.csv")
        #CREEDS_up_gene_sigs.to_csv(filename1) # csv for up genes
        #print(filename1)
        x2kCreedsDxDiseaseNames.append(filename1)
        
        CREEDS_down_gene_sigs = pd.DataFrame(CREEDS_down_genes[a])
        CREEDS_down_gene_sigs.columns = ["Genes", "Score"]
        CREEDS_disease_sig_down_genes_list = CREEDS_down_gene_sigs["Genes"]
        filename2 = (DOI + "_" + str(a) + "_CREEDS_disease_sig_down_genes.csv")
        #CREEDS_down_gene_sigs.to_csv(filename2) # csv for down genes
        
        ### X2K
        ## up genes
        CREEDS_X2K_up_genes = run_X2K(CREEDS_disease_sig_up_genes_list)
        CREEDS_X2K_up_genes = CREEDS_X2K_up_genes["X2K"]
        CREEDS_X2K_up_genes_df = pd.DataFrame(CREEDS_X2K_up_genes['nodes'])
        filename_up = (DOI + "_" + a + "_CREEDS_X2K_up_genes.csv")
        #CREEDS_X2K_up_genes_df.to_csv(filename_up) # THIS IS THE FILE THEY SHOULD BE ABLE TO DOWNLOAD
        ## down genes
        CREEDS_X2K_down_genes = run_X2K(CREEDS_disease_sig_down_genes_list)
        CREEDS_X2K_down_genes = CREEDS_X2K_down_genes["X2K"]
        CREEDS_X2K_down_genes_df = pd.DataFrame(CREEDS_X2K_down_genes['nodes'])
        filename_down = (DOI + "_" + a + "_CREEDS_X2K_down_genes.csv")
        #CREEDS_X2K_down_genes_df.to_csv(filename_down) # THIS IS THE FILE THEY SHOULD BE ABLE TO DOWNLOAD
    x2kCreedsDxNames_df = pd.DataFrame(x2kCreedsDxDiseaseNames)    
    return render_template('x2k_creeds_dx_diseases.html', x2k_creeds_sig_ids = CREEDS_disease_ids_df.to_json(), x2kCreedsDxDiseaseNames = x2kCreedsDxNames_df.to_json())

def return_x2k_creeds_dx_diseases_up_df(sig_id):
    CREEDS_up_gene_sigs = pd.DataFrame(CREEDS_up_genes[sig_id])
    CREEDS_up_gene_sigs.columns = ["Genes", "Score"]
    CREEDS_disease_sig_up_genes_list = CREEDS_up_gene_sigs["Genes"]
    #filename1 = (DOI + "_" + str(a) + "_CREEDS_disease_sig_up_genes.csv")
    #CREEDS_up_gene_sigs.to_csv(filename1) # csv for up genes
    #print(filename1)
    
    CREEDS_down_gene_sigs = pd.DataFrame(CREEDS_down_genes[sig_id])
    CREEDS_down_gene_sigs.columns = ["Genes", "Score"]
    CREEDS_disease_sig_down_genes_list = CREEDS_down_gene_sigs["Genes"]
    #filename2 = (DOI + "_" + str(a) + "_CREEDS_disease_sig_down_genes.csv")
    #CREEDS_down_gene_sigs.to_csv(filename2) # csv for down genes
    
    ### X2K
    ## up genes
    CREEDS_X2K_up_genes = run_X2K(CREEDS_disease_sig_up_genes_list)
    CREEDS_X2K_up_genes = CREEDS_X2K_up_genes["X2K"]
    CREEDS_X2K_up_genes_df = pd.DataFrame(CREEDS_X2K_up_genes['nodes'])
    #filename_up = (DOI + "_" + a + "_CREEDS_X2K_up_genes.csv")
    #CREEDS_X2K_up_genes_df.to_csv(filename_up) # THIS IS THE FILE THEY SHOULD BE ABLE TO DOWNLOAD
    ## down genes
    CREEDS_X2K_down_genes = run_X2K(CREEDS_disease_sig_down_genes_list)
    CREEDS_X2K_down_genes = CREEDS_X2K_down_genes["X2K"]
    CREEDS_X2K_down_genes_df = pd.DataFrame(CREEDS_X2K_down_genes['nodes'])
    #filename_down = (DOI + "_" + a + "_CREEDS_X2K_down_genes.csv")
    #CREEDS_X2K_down_genes_df.to_csv(filename_down) # THIS IS THE FILE THEY SHOULD BE ABLE TO DOWNLOAD
        
    return CREEDS_X2K_up_genes_df

def return_x2k_creeds_dx_diseases_down_df(sig_id):
    CREEDS_up_gene_sigs = pd.DataFrame(CREEDS_up_genes[sig_id])
    CREEDS_up_gene_sigs.columns = ["Genes", "Score"]
    CREEDS_disease_sig_up_genes_list = CREEDS_up_gene_sigs["Genes"]
    #filename1 = (DOI + "_" + str(a) + "_CREEDS_disease_sig_up_genes.csv")
    #CREEDS_up_gene_sigs.to_csv(filename1) # csv for up genes
    #print(filename1)
    
    CREEDS_down_gene_sigs = pd.DataFrame(CREEDS_down_genes[sig_id])
    CREEDS_down_gene_sigs.columns = ["Genes", "Score"]
    CREEDS_disease_sig_down_genes_list = CREEDS_down_gene_sigs["Genes"]
    #filename2 = (DOI + "_" + str(a) + "_CREEDS_disease_sig_down_genes.csv")
    #CREEDS_down_gene_sigs.to_csv(filename2) # csv for down genes
    
    ### X2K
    ## up genes
    CREEDS_X2K_up_genes = run_X2K(CREEDS_disease_sig_up_genes_list)
    CREEDS_X2K_up_genes = CREEDS_X2K_up_genes["X2K"]
    CREEDS_X2K_up_genes_df = pd.DataFrame(CREEDS_X2K_up_genes['nodes'])
    #filename_up = (DOI + "_" + a + "_CREEDS_X2K_up_genes.csv")
    #CREEDS_X2K_up_genes_df.to_csv(filename_up) # THIS IS THE FILE THEY SHOULD BE ABLE TO DOWNLOAD
    ## down genes
    CREEDS_X2K_down_genes = run_X2K(CREEDS_disease_sig_down_genes_list)
    CREEDS_X2K_down_genes = CREEDS_X2K_down_genes["X2K"]
    CREEDS_X2K_down_genes_df = pd.DataFrame(CREEDS_X2K_down_genes['nodes'])
    #filename_down = (DOI + "_" + a + "_CREEDS_X2K_down_genes.csv")
    #CREEDS_X2K_down_genes_df.to_csv(filename_down) # THIS IS THE FILE THEY SHOULD BE ABLE TO DOWNLOAD
        
          
    return CREEDS_X2K_down_genes_df


def return_x2k_L1000_diseases_up_df(sig_id, doi):
    print(sig_id)
    DOI_ICD9 = icd9_to_doid_final[icd9_to_doid_final.Disease.apply(lambda s: bool(re.compile(doi, re.IGNORECASE).search(s)))]
    DOI_ICD9_codes = DOI_ICD9.ICD9
    # get the do_id from the DOI
    DOI_DOID_codes = DOI_ICD9.DOID
    # get the do_id from the DOI
    DOI_CREEDS_codes = DOI_ICD9.CREEDS_drug_id
    print(doi)   
    ##filter by DOI. Need to convert DOI to ICD9 first.
    icd9_to_doid_final_search = icd9_to_doid_final[icd9_to_doid_final["Disease"].apply(lambda s: bool(re.compile(doi, re.IGNORECASE).search(s)))]
    icd9_to_doid_final_search1 = icd9_to_doid_final_search["ICD9"]
    ## rebuild the wildcard dataframe
    icd9_wildcard = pd.DataFrame(icd9_to_doid_final_search1, columns=['ICD9'])
    icd9_wildcard['ICD9_wildcard'] = icd9_wildcard['ICD9'].apply(lambda code: str(code).split('.')[0])
    icd9_wildcard.head()
    icd9_to_doid_final['ICD9_wildcard'] = icd9_to_doid_final['ICD9'].apply(lambda code: str(code).split('.')[0])
    ICD9_df_joined = pd.merge(
        left=icd9_wildcard, left_on='ICD9_wildcard',
        right=icd9_to_doid_final, right_on='ICD9_wildcard',
        how='inner',
        suffixes=(
            '_Manual',
            '_right',
        )
    )
    emr_sub = EMR_data_df[EMR_data_df["Description"].apply(lambda s: bool(re.compile(str(doi), re.IGNORECASE).search(str(s))))]
    emr_sub.reset_index(drop = True, inplace = True)
    emr_sub = []
    for a in ICD9_df_joined.ICD9_wildcard.unique():
        emr_sub1 = EMR_data_df[EMR_data_df["ICD9"].apply(lambda s: bool(re.compile(str(a), re.IGNORECASE).search(str(s))))]
        emr_sub.append(emr_sub1)
    emr_sub_df = pd.concat(emr_sub)
    top_drugs_from_disease = list(emr_sub_df.Drug_Name[0:5]) #take the top 5 drugs
    top_drugs_from_disease
    single_word_drugs = []
    for i in top_drugs_from_disease:
        j = str(i)
        splitted = j.split()
        first_word = splitted[0]
        single_word_drugs.append(first_word) 
    single_word_drugs = list(set(single_word_drugs))
    # Generate a blacklist process

    single_word_drugs_list = list(pd.Series(single_word_drugs).map(process_blacklist))
    single_word_drugs_list
    ## L1000 API Integration
    L1000FWD_URL = 'http://amp.pharm.mssm.edu/L1000FWD/'
    L1000_reverse_drugs_store = []
    L1000_reverse_pertids_store = []
    drug_and_perts = pd.DataFrame()
    for a in single_word_drugs_list:
        query_string = a
        L1000_reverse_drug_response = requests.get(L1000FWD_URL + 'synonyms/' + query_string)
        if L1000_reverse_drug_response.status_code == 200:
            #pprint(L1000_reverse_drug_response.json())
            print('a')
            L1000_reverse_significant_query = L1000_reverse_drug_response.json()
            if len(L1000_reverse_significant_query) > 0:
                #json.dump(L1000_reverse_drug_response.json(), open(query_string + '_L1000_reverse_drug_query.json', 'w'), indent=4)
                L1000_reverse_significant_query = L1000_reverse_drug_response.json()
                L1000_reverse_significant_query_df = pd.DataFrame(L1000_reverse_significant_query)
                L1000_reverse_pertids_store.append(list(L1000_reverse_significant_query_df["pert_id"]))
                
                # store the drug name with the pert ID
                drug_and_perts = drug_and_perts.append(L1000_reverse_significant_query_df)
                
                L1000_reverse_drugs_store.append(a)            
                print("Found significant L1000 drug signatures for " + query_string)            
            else:
                print("No significant L1000 drug signatures for " + query_string)     
    
        
    L1000_reverse_pertids_flat = []
    for sublist in L1000_reverse_pertids_store:
        for item in sublist:
            L1000_reverse_pertids_flat.append(item) 
    L1000_reverse_pertids_flat = set(list(L1000_reverse_pertids_flat)) & set(list(metadata["pert_id"])) & set(list(L1000_up_lookup.keys()))
    DrOI_disease_up_signatures = {k: L1000_up_lookup[k] for k in (L1000_reverse_pertids_flat)}
    DrOI_disease_up_no_perts = {k: v for d in DrOI_disease_up_signatures.values() for k, v in d.items()}
    DrOI_disease_up_drug_sigs = list(DrOI_disease_up_no_perts.keys())
    DrOI_disease_down_signatures = {k: L1000_down_lookup[k] for k in (L1000_reverse_pertids_flat)}
    DrOI_disease_down_no_perts = {k: v for d in DrOI_disease_down_signatures.values() for k, v in d.items()}
    DrOI_disease_down_drug_sigs = list(DrOI_disease_down_no_perts.keys())
    DrOI_disease_all_sigs = set(DrOI_disease_up_drug_sigs) & set(DrOI_disease_down_drug_sigs)
    for a in DrOI_disease_all_sigs:
        L1000_up_json_file = DrOI_disease_up_no_perts[a]
        L1000_down_json_file = DrOI_disease_down_no_perts[a]
        
        
        ## need to clean the creeds genes because they have p values attached
        ## X2K implementation
        ## up genes
        L1000_X2K_up_genes = run_X2K(L1000_up_json_file)
        L1000_X2K_up_genes = L1000_X2K_up_genes["X2K"]
        L1000_X2K_up_genes_df = pd.DataFrame(L1000_X2K_up_genes['nodes'])
        
        filename_up = (a + "_L1000_X2K_up_genes.csv")
        #L1000_X2K_up_genes_df.to_csv(filename_up) # THIS IS THE FILE THEY SHOULD BE ABLE TO DOWNLOAD
        print(filename_up)
        ## down genes
        L1000_X2K_down_genes = run_X2K(L1000_down_json_file)
        L1000_X2K_down_genes = L1000_X2K_down_genes["X2K"]
        L1000_X2K_down_genes_df = pd.DataFrame(L1000_X2K_down_genes['nodes'])
        
        filename_down = (a + "_L1000_X2K_down_genes.csv")
        #L1000_X2K_up_genes_df.to_csv(filename_up) # THIS IS THE FILE THEY SHOULD BE ABLE TO DOWNLOAD
        ### save Json files
        #with open(a + "_L1000_disease_up_sig.json", "w") as f:
            #json.dump(L1000_up_json_file, f)
    # with open(a + "_L1000_disease_down_sig.json", "w") as f:
            #json.dump(L1000_down_json_file, f)
    return L1000_X2K_up_genes_df


def return_x2k_L1000_diseases_down_df(sig_id, doi):
    print('a', sig_id)
    DOI_ICD9 = icd9_to_doid_final[icd9_to_doid_final.Disease.apply(lambda s: bool(re.compile(doi, re.IGNORECASE).search(s)))]
    DOI_ICD9_codes = DOI_ICD9.ICD9
    # get the do_id from the DOI
    DOI_DOID_codes = DOI_ICD9.DOID
    # get the do_id from the DOI
    DOI_CREEDS_codes = DOI_ICD9.CREEDS_drug_id
    print(doi)   
    ##filter by DOI. Need to convert DOI to ICD9 first.
    icd9_to_doid_final_search = icd9_to_doid_final[icd9_to_doid_final["Disease"].apply(lambda s: bool(re.compile(doi, re.IGNORECASE).search(s)))]
    icd9_to_doid_final_search1 = icd9_to_doid_final_search["ICD9"]
    ## rebuild the wildcard dataframe
    icd9_wildcard = pd.DataFrame(icd9_to_doid_final_search1, columns=['ICD9'])
    icd9_wildcard['ICD9_wildcard'] = icd9_wildcard['ICD9'].apply(lambda code: str(code).split('.')[0])
    icd9_wildcard.head()
    icd9_to_doid_final['ICD9_wildcard'] = icd9_to_doid_final['ICD9'].apply(lambda code: str(code).split('.')[0])
    ICD9_df_joined = pd.merge(
        left=icd9_wildcard, left_on='ICD9_wildcard',
        right=icd9_to_doid_final, right_on='ICD9_wildcard',
        how='inner',
        suffixes=(
            '_Manual',
            '_right',
        )
    )
    emr_sub = EMR_data_df[EMR_data_df["Description"].apply(lambda s: bool(re.compile(str(doi), re.IGNORECASE).search(str(s))))]
    emr_sub.reset_index(drop = True, inplace = True)
    emr_sub = []
    for a in ICD9_df_joined.ICD9_wildcard.unique():
        emr_sub1 = EMR_data_df[EMR_data_df["ICD9"].apply(lambda s: bool(re.compile(str(a), re.IGNORECASE).search(str(s))))]
        emr_sub.append(emr_sub1)
    emr_sub_df = pd.concat(emr_sub)
    top_drugs_from_disease = list(emr_sub_df.Drug_Name[0:5]) #take the top 5 drugs
    top_drugs_from_disease
    single_word_drugs = []
    for i in top_drugs_from_disease:
        j = str(i)
        splitted = j.split()
        first_word = splitted[0]
        single_word_drugs.append(first_word) 
    single_word_drugs = list(set(single_word_drugs))
    # Generate a blacklist process

    single_word_drugs_list = list(pd.Series(single_word_drugs).map(process_blacklist))
    single_word_drugs_list
    ## L1000 API Integration
    L1000FWD_URL = 'http://amp.pharm.mssm.edu/L1000FWD/'
    L1000_reverse_drugs_store = []
    L1000_reverse_pertids_store = []
    drug_and_perts = pd.DataFrame()
    for a in single_word_drugs_list:
        query_string = a
        L1000_reverse_drug_response = requests.get(L1000FWD_URL + 'synonyms/' + query_string)
        if L1000_reverse_drug_response.status_code == 200:
            #pprint(L1000_reverse_drug_response.json())
            print('a')
            L1000_reverse_significant_query = L1000_reverse_drug_response.json()
            if len(L1000_reverse_significant_query) > 0:
                #json.dump(L1000_reverse_drug_response.json(), open(query_string + '_L1000_reverse_drug_query.json', 'w'), indent=4)
                L1000_reverse_significant_query = L1000_reverse_drug_response.json()
                L1000_reverse_significant_query_df = pd.DataFrame(L1000_reverse_significant_query)
                L1000_reverse_pertids_store.append(list(L1000_reverse_significant_query_df["pert_id"]))
                
                # store the drug name with the pert ID
                drug_and_perts = drug_and_perts.append(L1000_reverse_significant_query_df)
                
                L1000_reverse_drugs_store.append(a)            
                print("Found significant L1000 drug signatures for " + query_string)            
            else:
                print("No significant L1000 drug signatures for " + query_string)     
    
        
    L1000_reverse_pertids_flat = []
    for sublist in L1000_reverse_pertids_store:
        for item in sublist:
            L1000_reverse_pertids_flat.append(item) 
    L1000_reverse_pertids_flat = set(list(L1000_reverse_pertids_flat)) & set(list(metadata["pert_id"])) & set(list(L1000_up_lookup.keys()))
    DrOI_disease_up_signatures = {k: L1000_up_lookup[k] for k in (L1000_reverse_pertids_flat)}
    DrOI_disease_up_no_perts = {k: v for d in DrOI_disease_up_signatures.values() for k, v in d.items()}
    DrOI_disease_up_drug_sigs = list(DrOI_disease_up_no_perts.keys())
    DrOI_disease_down_signatures = {k: L1000_down_lookup[k] for k in (L1000_reverse_pertids_flat)}
    DrOI_disease_down_no_perts = {k: v for d in DrOI_disease_down_signatures.values() for k, v in d.items()}
    DrOI_disease_down_drug_sigs = list(DrOI_disease_down_no_perts.keys())
    DrOI_disease_all_sigs = set(DrOI_disease_up_drug_sigs) & set(DrOI_disease_down_drug_sigs)
    for a in DrOI_disease_all_sigs:
        L1000_up_json_file = DrOI_disease_up_no_perts[a]
        L1000_down_json_file = DrOI_disease_down_no_perts[a]
        
        
        ## need to clean the creeds genes because they have p values attached
        ## X2K implementation
        ## up genes
        L1000_X2K_up_genes = run_X2K(L1000_up_json_file)
        L1000_X2K_up_genes = L1000_X2K_up_genes["X2K"]
        L1000_X2K_up_genes_df = pd.DataFrame(L1000_X2K_up_genes['nodes'])
        
        filename_up = (a + "_L1000_X2K_up_genes.csv")
        #L1000_X2K_up_genes_df.to_csv(filename_up) # THIS IS THE FILE THEY SHOULD BE ABLE TO DOWNLOAD
        print(filename_up)
        ## down genes
        L1000_X2K_down_genes = run_X2K(L1000_down_json_file)
        L1000_X2K_down_genes = L1000_X2K_down_genes["X2K"]
        L1000_X2K_down_genes_df = pd.DataFrame(L1000_X2K_down_genes['nodes'])
        
        filename_down = (a + "_L1000_X2K_down_genes.csv")
        #L1000_X2K_up_genes_df.to_csv(filename_up) # THIS IS THE FILE THEY SHOULD BE ABLE TO DOWNLOAD
        ### save Json files
        #with open(a + "_L1000_disease_up_sig.json", "w") as f:
            #json.dump(L1000_up_json_file, f)
    # with open(a + "_L1000_disease_down_sig.json", "w") as f:
            #json.dump(L1000_down_json_file, f)
    return L1000_X2K_down_genes_df

def return_up_df(sig_id):
    L1000FWD_URL = 'http://amp.pharm.mssm.edu/L1000FWD/'
    L1000_gene_response = requests.get(L1000FWD_URL + 'sig/' + str(sig_id))
    if L1000_gene_response.status_code == 200:
                #pprint(response.json()
        L1000_query = L1000_gene_response.json() 
        L1000_query_up_genes = L1000_query["up_genes"]
        L1000_query_down_genes = L1000_query["down_genes"]
                
                
                ### UNIQUE X2K CODE
                
                #L1000 up genes
    L1000_X2K_up_genes = run_X2K(L1000_query_up_genes)
    L1000_X2K_up_genes = L1000_X2K_up_genes["X2K"]
    L1000_X2K_up_genes_df = pd.DataFrame(L1000_X2K_up_genes['nodes'])       
    return L1000_X2K_up_genes_df

def return_down_df(sig_id):
    L1000FWD_URL = 'http://amp.pharm.mssm.edu/L1000FWD/'
    L1000_gene_response = requests.get(L1000FWD_URL + 'sig/' + str(sig_id))
    if L1000_gene_response.status_code == 200:
                #pprint(response.json()
        L1000_query = L1000_gene_response.json() 
        L1000_query_up_genes = L1000_query["up_genes"]
        L1000_query_down_genes = L1000_query["down_genes"]

    L1000_X2K_down_genes = run_X2K(L1000_query_down_genes)
    L1000_X2K_down_genes = L1000_X2K_down_genes["X2K"]
    L1000_X2K_down_genes_df = pd.DataFrame(L1000_X2K_down_genes['nodes'])
    return L1000_X2K_down_genes_df

def return_creeds_diseases_up(sig_id):
    CREEDS_URL = 'http://amp.pharm.mssm.edu/CREEDS/'
    CREEEDS_Disease_response = requests.get(CREEDS_URL + 'api', params={'id':str(sig_id)})
    if CREEEDS_Disease_response.status_code == 200:
        #pprint(CREEEDS_Disease_response.json())
        #json.dump(CREEEDS_Drug_response.json(), open(CREEDS_drug_final_GSE_ID[a] + '_api1_result.json', 'w'), indent=4)
        CREEEDS_Disease_response_json = CREEEDS_Disease_response.json()
        
        ## up genes
        CREEDS_disease_sigs_up_genes = CREEEDS_Disease_response_json['up_genes']
        CREEDS_disease_sigs_up_genes_df = pd.DataFrame(CREEDS_disease_sigs_up_genes) # this is the up genes dataframe
        CREEDS_disease_sigs_up_genes_df.columns = ["Gene_Name", "Score"]
        #filename1 = (str(CREEDS_drug_ids_list[a]) + "_CREEDS_disease_sig_up_genes.csv")
        #CREEDS_disease_sigs_up_genes_df.to_csv(filename1) # this saves the df as a csv
        
        
         # this is the up genes dataframe
    return CREEDS_disease_sigs_up_genes_df



def return_creeds_diseases_down(sig_id):
    CREEDS_URL = 'http://amp.pharm.mssm.edu/CREEDS/'
    CREEEDS_Disease_response = requests.get(CREEDS_URL + 'api', params={'id':str(sig_id)})
    if CREEEDS_Disease_response.status_code == 200:
        #pprint(CREEEDS_Disease_response.json())
        #json.dump(CREEEDS_Drug_response.json(), open(CREEDS_drug_final_GSE_ID[a] + '_api1_result.json', 'w'), indent=4)
        CREEEDS_Disease_response_json = CREEEDS_Disease_response.json()

        ## down genes
        CREEDS_disease_sigs_down_genes = CREEEDS_Disease_response_json['down_genes']
        CREEDS_disease_sigs_down_genes_df = pd.DataFrame(CREEDS_disease_sigs_down_genes) 
        CREEDS_disease_sigs_down_genes_df.columns = ["Gene", "Score"]
        
    return CREEDS_disease_sigs_down_genes_df


def return_creeds_drugs_up(sig_id):
    CREEDS_URL = 'http://amp.pharm.mssm.edu/CREEDS/'
    CREEDS_drug_sigs_response = requests.get(CREEDS_URL + 'api', params={'id':str(sig_id)})
    if CREEDS_drug_sigs_response.status_code == 200:
        CREEDS_drug_sigs_response_json = CREEDS_drug_sigs_response.json()
            
            ## up genes
        CREEDS_drug_sigs_up_genes = CREEDS_drug_sigs_response_json['up_genes']
        CREEDS_drug_sigs_up_genes_df = pd.DataFrame(CREEDS_drug_sigs_up_genes)
        CREEDS_drug_sigs_up_genes_df.columns = ['name', 'pvalue']
    return CREEDS_drug_sigs_up_genes_df

def return_creeds_drugs_down(sig_id):
    CREEDS_URL = 'http://amp.pharm.mssm.edu/CREEDS/'
    CREEDS_drug_sigs_response = requests.get(CREEDS_URL + 'api', params={'id':str(sig_id)})
    if CREEDS_drug_sigs_response.status_code == 200:
        CREEDS_drug_sigs_response_json = CREEDS_drug_sigs_response.json()
            
            ## up genes
        CREEDS_drug_sigs_down_genes = CREEDS_drug_sigs_response_json['down_genes']
        CREEDS_drug_sigs_down_genes_df = pd.DataFrame(CREEDS_drug_sigs_down_genes)
        CREEDS_drug_sigs_down_genes_df.columns = ['name', 'pvalue']
    return CREEDS_drug_sigs_down_genes_df

def return_x2k_creeds_diseases_up_df(sig_id):
    ### CREEDS DRUG CARD 


#for a in loop_iteration:
    CREEDS_URL = 'http://amp.pharm.mssm.edu/CREEDS/'
#     CREEEDS_Drug_response = requests.get(CREEDS_URL + 'search', params={'q':DrOI})
#     if CREEEDS_Drug_response.status_code == 200:
#     #pprint(CREEEDS_Drug_response.json())
#     #json.dump(CREEEDS_Drug_response.json(), open(DrOI + '_api1_result.json', 'w'), indent=4)
#         CREEDS_drug_output_df = pd.DataFrame(CREEEDS_Drug_response.json())

# #CREEDS_drug_output_df #Display this table
#     CREEDS_drug_sigs = list(CREEDS_drug_output_df.id)

#     for a in CREEDS_drug_sigs:
    CREEDS_X2K_drug_response = requests.get(CREEDS_URL + 'api', params={'id':str(sig_id)})
    if CREEDS_X2K_drug_response.status_code == 200:
        #pprint(CREEDS_X2K_drug_response.json())
        #json.dump(CREEDS_X2K_drug_response.json(), open('api3_result.json', 'wb'), indent=4)
        CREEDS_signatures_total = CREEDS_X2K_drug_response.json()
        
        
        CREEDS_query_up_genes = CREEDS_signatures_total["up_genes"]
        CREEDS_query_up_genes = pd.DataFrame(CREEDS_query_up_genes)
        CREEDS_query_up_genes.columns = ["Up_genes", "Score"]
        CREEDS_query_up_genes = list(CREEDS_query_up_genes["Up_genes"])
        
        CREEDS_query_down_genes = CREEDS_signatures_total["down_genes"]
        CREEDS_query_down_genes = pd.DataFrame(CREEDS_query_down_genes)
        CREEDS_query_down_genes.columns = ["Down_genes", "Score"]
        CREEDS_query_down_genes = list(CREEDS_query_down_genes["Down_genes"])
        
        ## need to clean the creeds genes because they have p values attached
        
        
        ## X2K implementation
        
        ## up genes
        CREEDS_X2K_up_genes = run_X2K(CREEDS_query_up_genes)
        CREEDS_X2K_up_genes = CREEDS_X2K_up_genes["X2K"]
        CREEDS_X2K_up_genes_df = pd.DataFrame(CREEDS_X2K_up_genes['nodes'])
        # filename_up = (a + "_CREEDS_X2K_up_genes.csv")
            # CREEDS_X2K_up_genes_df.to_csv(filename_up) # THIS IS THE FILE THEY SHOULD BE ABLE TO DOWNLOAD
        
        ## down genes
        CREEDS_X2K_down_genes = run_X2K(CREEDS_query_down_genes)
        CREEDS_X2K_down_genes = CREEDS_X2K_down_genes["X2K"]
        CREEDS_X2K_down_genes_df = pd.DataFrame(CREEDS_X2K_down_genes['nodes'])
        # filename_down = (a + "_CREEDS_X2K_down_genes.csv")
            # CREEDS_X2K_down_genes_df.to_csv(filename_down) # THIS IS THE FILE THEY SHOULD BE ABLE TO DOWNLOAD
        
    return CREEDS_X2K_up_genes_df

def return_x2k_creeds_diseases_down_df(sig_id):
    CREEDS_URL = 'http://amp.pharm.mssm.edu/CREEDS/'
#     CREEEDS_Drug_response = requests.get(CREEDS_URL + 'search', params={'q':DrOI})
#     if CREEEDS_Drug_response.status_code == 200:
#     #pprint(CREEEDS_Drug_response.json())
#     #json.dump(CREEEDS_Drug_response.json(), open(DrOI + '_api1_result.json', 'w'), indent=4)
#         CREEDS_drug_output_df = pd.DataFrame(CREEEDS_Drug_response.json())

# #CREEDS_drug_output_df #Display this table
#     CREEDS_drug_sigs = list(CREEDS_drug_output_df.id)

#     for a in CREEDS_drug_sigs:
    CREEDS_X2K_drug_response = requests.get(CREEDS_URL + 'api', params={'id':str(sig_id)})
    if CREEDS_X2K_drug_response.status_code == 200:
        #pprint(CREEDS_X2K_drug_response.json())
        #json.dump(CREEDS_X2K_drug_response.json(), open('api3_result.json', 'wb'), indent=4)
        CREEDS_signatures_total = CREEDS_X2K_drug_response.json()
        
        
        CREEDS_query_up_genes = CREEDS_signatures_total["up_genes"]
        CREEDS_query_up_genes = pd.DataFrame(CREEDS_query_up_genes)
        CREEDS_query_up_genes.columns = ["Up_genes", "Score"]
        CREEDS_query_up_genes = list(CREEDS_query_up_genes["Up_genes"])
        
        CREEDS_query_down_genes = CREEDS_signatures_total["down_genes"]
        CREEDS_query_down_genes = pd.DataFrame(CREEDS_query_down_genes)
        CREEDS_query_down_genes.columns = ["Down_genes", "Score"]
        CREEDS_query_down_genes = list(CREEDS_query_down_genes["Down_genes"])

        CREEDS_X2K_down_genes = run_X2K(CREEDS_query_down_genes)
        CREEDS_X2K_down_genes = CREEDS_X2K_down_genes["X2K"]
        CREEDS_X2K_down_genes_df = pd.DataFrame(CREEDS_X2K_down_genes['nodes'])
        # filename_down = (a + "_CREEDS_X2K_down_genes.csv")

    return CREEDS_X2K_down_genes_df

@app.route(ENDPOINT + '/download_creeds_diseases_upgenes/<string:name>')
def download_creeds_diseases_up_csv(name):
    proxy = io.StringIO()
    return_creeds_diseases_up(name).to_csv(proxy)
    temp_file = io.BytesIO()
    temp_file.write(proxy.getvalue().encode('utf-8'))
    temp_file.seek(0)
    proxy.close()
    return send_file(temp_file, attachment_filename="%s_upgenes_creeds_diseases.csv"%name, as_attachment=True)

@app.route(ENDPOINT + '/download_creeds_diseases_downgenes/<string:name>')
def download_creeds_diseases_down_csv(name):
    proxy = io.StringIO()
    return_creeds_diseases_down(name).to_csv(proxy)
    temp_file = io.BytesIO()
    temp_file.write(proxy.getvalue().encode('utf-8'))
    temp_file.seek(0)
    proxy.close()
    return send_file(temp_file, attachment_filename="%s_downgenes_creeds_diseases.csv"%name, as_attachment=True)
    

@app.route(ENDPOINT + '/download_creeds_drugs_upgenes/<string:name>')
def download_creeds_up_csv(name):
    proxy = io.StringIO()
    return_creeds_drugs_up(name).to_csv(proxy)
    temp_file = io.BytesIO()
    temp_file.write(proxy.getvalue().encode('utf-8'))
    temp_file.seek(0)
    proxy.close()
    return send_file(temp_file, attachment_filename="%s_upgenes_creeds.csv"%name, as_attachment=True)

@app.route(ENDPOINT + '/download_creeds_drugs_downgenes/<string:name>')
def download_creeds_down_csv(name):
    proxy = io.StringIO()
    return_creeds_drugs_down(name).to_csv(proxy)
    temp_file = io.BytesIO()
    temp_file.write(proxy.getvalue().encode('utf-8'))
    temp_file.seek(0)
    proxy.close()
    return send_file(temp_file, attachment_filename="%s_downgenes_creeds.csv"%name, as_attachment=True)


@app.route(ENDPOINT + '/download_x2k_creeds_dx_diseases_upgenes/<string:name>')
def download_x2k_creeds_up_csv(name):
    proxy = io.StringIO()
    return_x2k_creeds_dx_diseases_up_df(name).to_csv(proxy)
    temp_file = io.BytesIO()
    temp_file.write(proxy.getvalue().encode('utf-8'))
    temp_file.seek(0)
    proxy.close()
    return send_file(temp_file, attachment_filename="%s_upgenes_x2k_creeds.csv"%name, as_attachment=True)

@app.route(ENDPOINT + '/download_x2k_creeds_dx_diseases_downgenes/<string:name>')
def download_x2k_creeds_down_csv(name):
    proxy = io.StringIO()
    return_x2k_creeds_dx_diseases_down_df(name).to_csv(proxy)
    temp_file = io.BytesIO()
    temp_file.write(proxy.getvalue().encode('utf-8'))
    temp_file.seek(0)
    proxy.close()
    return send_file(temp_file, attachment_filename="%s_downgenes_x2k_creeds.csv"%name, as_attachment=True)

@app.route(ENDPOINT + '/download_x2k_l1000_upgenes/<string:name>')
def download_x2k_up_csv(name):
    proxy = io.StringIO()
    return_up_df(name).to_csv(proxy)
    temp_file = io.BytesIO()
    temp_file.write(proxy.getvalue().encode('utf-8'))
    temp_file.seek(0)
    proxy.close()
    return send_file(temp_file, attachment_filename="%s_upgenes.csv"%name, as_attachment=True)

@app.route(ENDPOINT + '/download_x2k_l1000_downgenes/<string:name>')
def download_x2k_down_csv(name):
    proxy = io.StringIO()
    return_down_df(name).to_csv(proxy)
    temp_file = io.BytesIO()
    temp_file.write(proxy.getvalue().encode('utf-8'))
    temp_file.seek(0)
    proxy.close()
    return send_file(temp_file, attachment_filename="%s_downgenes.csv"%name, as_attachment=True)

@app.route(ENDPOINT + '/download_x2k_creeds_rx_diseases_upgenes/<string:name>')
def download_x2k_creeds_rx_diseases_up_csv(name):
    proxy = io.StringIO()
    return_x2k_creeds_rx_diseases_up_df(name).to_csv(proxy)
    temp_file = io.BytesIO()
    temp_file.write(proxy.getvalue().encode('utf-8'))
    temp_file.seek(0)
    proxy.close()
    return send_file(temp_file, attachment_filename="%s_upgenes_x2k_creeds_rx_diseases.csv"%name, as_attachment=True)

@app.route(ENDPOINT + '/download_x2k_creeds_rx_diseases_downgenes/<string:name>')
def download_x2k_creeds_rx_diseases_down_csv(name):
    proxy = io.StringIO()
    return_x2k_creeds_rx_diseases_down_df(name).to_csv(proxy)
    temp_file = io.BytesIO()
    temp_file.write(proxy.getvalue().encode('utf-8'))
    temp_file.seek(0)
    proxy.close()
    return send_file(temp_file, attachment_filename="%s_downgenes_x2k_creeds_rx_diseases.csv"%name, as_attachment=True)


@app.route(ENDPOINT + '/download_creeds_dx_dx_upgenes/<string:name>')
def download_creeds_dx_dx_up_csv(name):
    proxy = io.StringIO()
    return_creeds_dx_dx_up_df(name).to_csv(proxy)
    temp_file = io.BytesIO()
    temp_file.write(proxy.getvalue().encode('utf-8'))
    temp_file.seek(0)
    proxy.close()
    return send_file(temp_file, attachment_filename="%s_creeds_dx_dx_upgenes.csv"%name, as_attachment=True)
    
@app.route(ENDPOINT + '/download_creeds_dx_dx_downgenes/<string:name>')
def download_creeds_dx_dx_down_csv(name):
    proxy = io.StringIO()
    return_creeds_dx_dx_down_df(name).to_csv(proxy)
    temp_file = io.BytesIO()
    temp_file.write(proxy.getvalue().encode('utf-8'))
    temp_file.seek(0)
    proxy.close()
    return send_file(temp_file, attachment_filename="%s_creeds_dx_dx_downgenes.csv"%name, as_attachment=True)

@app.route(ENDPOINT + '/download_x2k_L1000_diseases_upgenes/<string:name>/<string:doi>')
def download_x2k_L1000_diseases_upgenes(name, doi):
    proxy = io.StringIO()
    return_x2k_L1000_diseases_up_df(name, doi).to_csv(proxy)
    temp_file = io.BytesIO()
    temp_file.write(proxy.getvalue().encode('utf-8'))
    temp_file.seek(0)
    proxy.close()
    return send_file(temp_file, attachment_filename="%s_x2k_L1000_diseases_upgenes.csv"%name, as_attachment=True)

@app.route(ENDPOINT + '/download_x2k_L1000_diseases_downgenes/<string:name>/<string:doi>')
def download_x2k_L1000_diseases_downgenes(name, doi):
    proxy = io.StringIO()
    return_x2k_L1000_diseases_down_df(name, doi).to_csv(proxy)
    temp_file = io.BytesIO()
    temp_file.write(proxy.getvalue().encode('utf-8'))
    temp_file.seek(0)
    proxy.close()
    return send_file(temp_file, attachment_filename="%s_x2k_L1000_diseases_downgenes.csv"%name, as_attachment=True)

@app.route(ENDPOINT + '/download_drug_matrix_drugs_upgenes/<string:name>/<string:droi>')
def download_drug_matrix_drugs_upgenes(name, droi):
    proxy = io.StringIO()
    return_drugMatrix_up_df(name, droi).to_csv(proxy)
    temp_file = io.BytesIO()
    temp_file.write(proxy.getvalue().encode('utf-8'))
    temp_file.seek(0)
    proxy.close()
    return send_file(temp_file, attachment_filename="%s_drug_matrix_drugs_upgenes.csv"%name, as_attachment=True)

@app.route(ENDPOINT + '/download_drug_matrix_drugs_downgenes/<string:name>/<string:droi>')
def download_drug_matrix_drugs_downgenes(name, droi):
    proxy = io.StringIO()
    return_drugMatrix_down_df(name, droi).to_csv(proxy)
    temp_file = io.BytesIO()
    temp_file.write(proxy.getvalue().encode('utf-8'))
    temp_file.seek(0)
    proxy.close()
    return send_file(temp_file, attachment_filename="%s_drug_matrix_drugs_downgenes.csv"%name, as_attachment=True)

@app.route(ENDPOINT + '/download_drug_matrix_diseases_upgenes/<string:name>/<string:doi>')
def download_drug_matrix_diseases_upgenes(name, doi):
    proxy = io.StringIO()
    return_drugMatrix_diseases_up_df(name, doi).to_csv(proxy)
    temp_file = io.BytesIO()
    temp_file.write(proxy.getvalue().encode('utf-8'))
    temp_file.seek(0)
    proxy.close()
    return send_file(temp_file, attachment_filename="%s_drug_matrix_diseases_upgenes.csv"%name, as_attachment=True)

@app.route(ENDPOINT + '/download_drug_matrix_diseases_downgenes/<string:name>/<string:doi>')
def download_drug_matrix_diseases_downgenes(name, doi):
    proxy = io.StringIO()
    return_drugMatrix_diseases_down_df(name, doi).to_csv(proxy)
    temp_file = io.BytesIO()
    temp_file.write(proxy.getvalue().encode('utf-8'))
    temp_file.seek(0)
    proxy.close()
    return send_file(temp_file, attachment_filename="%s_drug_matrix_diseases_downgenes.csv"%name, as_attachment=True)

@app.route(ENDPOINT + '/download_creeds_rx_dx_upgenes/<string:name>')
def download_creeds_rx_dx_upgenes(name):
    proxy = io.StringIO()
    return_creeds_rx_dx_up_df(name).to_csv(proxy)
    temp_file = io.BytesIO()
    temp_file.write(proxy.getvalue().encode('utf-8'))
    temp_file.seek(0)
    proxy.close()
    return send_file(temp_file, attachment_filename="%s_creeds_rx_dx_upgenes.csv"%name, as_attachment=True)

@app.route(ENDPOINT + '/download_creeds_rx_dx_downgenes/<string:name>')
def download_creeds_rx_dx_downgenes(name):
    proxy = io.StringIO()
    return_creeds_rx_dx_down_df(name).to_csv(proxy)
    temp_file = io.BytesIO()
    temp_file.write(proxy.getvalue().encode('utf-8'))
    temp_file.seek(0)
    proxy.close()
    return send_file(temp_file, attachment_filename="%s_creeds_rx_dx_downgenes.csv"%name, as_attachment=True)

@app.route(ENDPOINT + '/download_x2k_creeds_diseases_upgenes/<string:name>')
def download_x2k_creeds_diseases_upgenes(name):
    proxy = io.StringIO()
    return_x2k_creeds_diseases_up_df(name).to_csv(proxy)
    temp_file = io.BytesIO()
    temp_file.write(proxy.getvalue().encode('utf-8'))
    temp_file.seek(0)
    proxy.close()
    return send_file(temp_file, attachment_filename="%s_x2k_creeds_diseases_up.csv"%name, as_attachment=True)

@app.route(ENDPOINT + '/download_x2k_creeds_diseases_downgenes/<string:name>')
def download_x2k_creeds_diseases_downgenes(name):
    proxy = io.StringIO()
    return_x2k_creeds_diseases_down_df(name).to_csv(proxy)
    temp_file = io.BytesIO()
    temp_file.write(proxy.getvalue().encode('utf-8'))
    temp_file.seek(0)
    proxy.close()
    return send_file(temp_file, attachment_filename="%s_x2k_creeds_diseases_down.csv"%name, as_attachment=True)

@app.route(ENDPOINT + '/download_x2k_creeds_dx_drugs_upgenes/<string:name>')
def download_x2k_creeds_dx_drugs_upgenes(name):
    proxy = io.StringIO()
    return_x2k_creeds_dx_drugs_up_df(name).to_csv(proxy)
    temp_file = io.BytesIO()
    temp_file.write(proxy.getvalue().encode('utf-8'))
    temp_file.seek(0)
    proxy.close()
    return send_file(temp_file, attachment_filename="%s_x2k_creeds_dx_drugs_up.csv"%name, as_attachment=True)


@app.route(ENDPOINT + '/download_x2k_creeds_dx_drugs_downgenes/<string:name>')
def download_x2k_creeds_dx_drugs_downgenes(name):
    proxy = io.StringIO()
    return_x2k_creeds_dx_drugs_up_df(name).to_csv(proxy)
    temp_file = io.BytesIO()
    temp_file.write(proxy.getvalue().encode('utf-8'))
    temp_file.seek(0)
    proxy.close()
    return send_file(temp_file, attachment_filename="%s_x2k_creeds_dx_drugs_up.csv"%name, as_attachment=True)

@app.route(ENDPOINT + '/clustergrammer', methods=['GET', 'POST'])
def clustergrammer():
    ### L1000
    # data = request.get_json(force=True)
    # DrOI = data['input']
### LOAD IN DRUG MATRIX FILE
# request.args.get('nm')
    DrOI = request.args.get('drug')
    print(DrOI)
    # DrOI = "acarbose"
    DrugMatrix ={}
    for line in urllib.request.urlopen('http://amp.pharm.mssm.edu/lincs-playground/index.php/s/bl6rC50ALsbXfw1/download'):
        label, genelist = line.decode().split('\t\t', maxsplit=1) 
        genelist_split = genelist.strip().split('\t')
        DrugMatrix[label] = genelist_split
    DrugMatrix = {x.replace('/', '_'): v  
        for x, v in DrugMatrix.items()}
    ### DrugMatrix drug input
    ## generate a list of searchable keys to reduce dict
    DrugMatrix_keys = pd.DataFrame(list(DrugMatrix.keys()))
    DrugMatrix_keys.columns = ["sigs"]
    Drug_Matrix_DrOI = DrugMatrix_keys[DrugMatrix_keys["sigs"].apply(lambda s: bool(re.compile(str(DrOI), re.IGNORECASE).search(str(s))))]
    ## reduce dict
    Drug_matrix_sigs_reduced = list(Drug_Matrix_DrOI["sigs"])
    #DrugMatrix_sigs = {k: DrugMatrix[k] for k in list(Drug_matrix_sigs_reduced["sigs"])} # total sigs
    ## up sigs
    Drug_matrix_up_sigs_reduced = Drug_Matrix_DrOI[Drug_Matrix_DrOI["sigs"].apply(lambda s: bool(re.compile(str("-up"), re.IGNORECASE).search(str(s))))]
    DrugMatrix_up_sigs= {k: DrugMatrix[k] for k in list(Drug_matrix_up_sigs_reduced["sigs"])}
    for a in list(Drug_matrix_up_sigs_reduced["sigs"]):
        DrugMatrix_up_sigs_save = DrugMatrix_up_sigs[a]
        print(a)
        #with open(a + "_DrugMatrix_up_sig.json", "w") as f:
            #json.dump(DrugMatrix_up_sigs_save, f)
            
    ## down sigs
    Drug_matrix_down_sigs_reduced = Drug_Matrix_DrOI[Drug_Matrix_DrOI["sigs"].apply(lambda s: bool(re.compile(str("-dn"), re.IGNORECASE).search(str(s))))]
    DrugMatrix_down_sigs= {k: DrugMatrix[k] for k in list(Drug_matrix_down_sigs_reduced["sigs"])}
    for b in list(Drug_matrix_down_sigs_reduced["sigs"]):
        DrugMatrix_down_sigs_save = DrugMatrix_down_sigs[b]
        print(b)
        #with open(b + "_DrugMatrix_down_sig.json", "w") as f:
            #json.dump(DrugMatrix_down_sigs_save, f)
            
            
    ### L1000 DRUG CARD DRUG INPUT
    DrOI_df = metadata[metadata["pert_desc"] == DrOI]
    DrOI_pert_ids = list(DrOI_df["pert_id"])
    DrOI_up_signatures = {k: L1000_up_lookup[k] for k in (DrOI_pert_ids)}
    DrOI_up_no_perts = {k: v for d in DrOI_up_signatures.values() for k, v in d.items()}
    DrOI_up_drug_sigs = list(DrOI_up_no_perts.keys())
    DrOI_down_signatures = {k: L1000_down_lookup[k] for k in (DrOI_pert_ids)}
    DrOI_down_no_perts = {k: v for d in DrOI_down_signatures.values() for k, v in d.items()}
    DrOI_down_drug_sigs = list(DrOI_down_no_perts.keys())
    DrOI_all_sigs = set(DrOI_up_drug_sigs) & set (DrOI_down_drug_sigs)
    DrOI_all_sigs_up = [s + "_up" for s in DrOI_all_sigs]
    DrOI_all_sigs_down = [s + "_down" for s in DrOI_all_sigs]
    ######## NEW CODE
    DrOI_all_sigs_display = [DrOI + "_" + s for s in list(DrOI_all_sigs)]
    ########
    for a in DrOI_all_sigs:
        L1000_up_json_file = DrOI_up_no_perts[a]
        L1000_down_json_file = DrOI_down_no_perts[a]
        print(a)
        #with open(a + "_L1000_up_sig.json", "w") as f:
            #json.dump(L1000_up_json_file, f)
        #with open(a + "_L1000_down_sig.json", "w") as f:
            #json.dump(L1000_down_json_file, f)
            
    ### CREEDS DRUG CARD 
    #for a in loop_iteration:
    CREEDS_URL = 'http://amp.pharm.mssm.edu/CREEDS/'
    CREEEDS_Drug_response = requests.get(CREEDS_URL + 'search', params={'q':DrOI})
    if CREEEDS_Drug_response.status_code == 200:
        #pprint(CREEEDS_Drug_response.json())
        #json.dump(CREEEDS_Drug_response.json(), open(DrOI + '_api1_result.json', 'w'), indent=4)
        CREEDS_drug_output_df = pd.DataFrame(CREEEDS_Drug_response.json())
        CREEDS_drug_output_ids = list(CREEDS_drug_output_df["id"])
        
        CREEDS_drug_output_ids_up = ["CREEDS_" + s + "_up" for s in CREEDS_drug_output_ids]
        CREEDS_drug_output_ids_down = ["CREEDS_" + s + "_down" for s in CREEDS_drug_output_ids]
        
        CREEDS_all_down_genes = []
        CREEDS_all_up_genes = []
        CREEDS_desc = []
        for a in CREEDS_drug_output_ids:
            CREEDS_URL = 'http://amp.pharm.mssm.edu/CREEDS/'
            CREEDS_drug_sigs_response = requests.get(CREEDS_URL + 'api', params={'id':str(a)})
            if CREEDS_drug_sigs_response.status_code == 200:
                CREEDS_drug_sigs_response_json = CREEDS_drug_sigs_response.json()
                
                ## up genes
                CREEDS_drug_sigs_up_genes = CREEDS_drug_sigs_response_json['up_genes']
                CREEDS_drug_sigs_up_genes_df = pd.DataFrame(CREEDS_drug_sigs_up_genes) # this is the up genes dataframe
                CREEDS_drug_sigs_up_genes_df.columns = ["Genes", "Score"]
                filename1 = (a + "_CREEDS_drug_sig_up_genes.csv")
                #CREEDS_drug_sigs_up_genes_df.to_csv(filename1) # this saves the df as a csv
                desc = (a + "_" + DrOI + "_" + CREEDS_drug_sigs_response_json["geo_id"])
                CREEDS_desc.append(desc)
                CREEDS_all_up_genes.append(list(CREEDS_drug_sigs_up_genes_df["Genes"]))
                
                ## down genes
                CREEDS_drug_sigs_down_genes = CREEDS_drug_sigs_response_json['down_genes']
                CREEDS_drug_sigs_down_genes_df = pd.DataFrame(CREEDS_drug_sigs_down_genes)# this is the down genes dataframe
                CREEDS_drug_sigs_down_genes_df.columns = ["Genes", "Score"]
                filename2 = (a + "_CREEDS_drug_sig_down_genes.csv")
                CREEDS_all_down_genes.append(list(CREEDS_drug_sigs_down_genes_df["Genes"]))
                #CREEDS_drug_sigs_down_genes_df.to_csv(filename2)
                #CREEDS_drug_sigs_down_genes = CREEDS_drug_sigs_response_json['down_genes'] # this saves the df as a csv
                print(filename2)
                
                up_keys = ['up_genes']
                gene_dict_up = {x:CREEDS_drug_sigs_response_json[x] for x in up_keys}
                gene_dict_up = {"CREEDS_" + a + "_" + k: v for k, v in gene_dict_up.items()}
                
                down_keys = ['down_genes']
                gene_dict_down = {x:CREEDS_drug_sigs_response_json[x] for x in down_keys}
                gene_dict_down = {"CREEDS_" + a + "_" + k: v for k, v in gene_dict_down.items()}
                
#storing genes in gene sets and signatures 
    ### CREEDS DISEASE CARD (DRUG INPUT)
    # RETURNS THE do_id, geo_id, and disease name in a dictionary
    CREEDS_GSE = {
        row['id']: [row['geo_id'], row["disease_name"]]
        for row in CREEDS_data
    }
    ## filter by DrOI need icd9 codes for proper conversion and query through CREEDS
    droi_search =EMR_data_df[EMR_data_df['Drug_Name'].apply(lambda s: bool(re.compile(DrOI, re.IGNORECASE).search(s)))]
    droi_search_top5 = droi_search[0:10]
    EMR_top_disease_from_drug = droi_search_top5["ICD9"]
    #top_disease_from_drug = EMR_top_disease_from_drug[0:5]
    ## build a datatable of all the ICD-9 CM diagnosis codes families (i.e no decimal points)
    EMR_top_disease_from_drug_df = pd.DataFrame(EMR_top_disease_from_drug, columns=['ICD9'])
    EMR_top_disease_from_drug_df['ICD9_wildcard'] = EMR_top_disease_from_drug_df['ICD9'].apply(lambda code: code.split('.')[0])
    #EMR_top_disease_from_drug_df.head()
    icd9_to_doid_final['ICD9_wildcard'] = icd9_to_doid_final['ICD9'].apply(lambda code: str(code).split('.')[0])
    #icd9_to_doid_final.head()
    df_joined = pd.merge(
        left=EMR_top_disease_from_drug_df, left_on='ICD9_wildcard',
        right=icd9_to_doid_final, right_on='ICD9_wildcard',
        how='inner',
        suffixes=(
            '_left',
            '_right',
        )
    )
    CREEDS_drug_ids = pd.DataFrame(set(df_joined.CREEDS_drug_id))
    CREEDS_drug_ids_list = list(set(df_joined.CREEDS_drug_id))
    #CREEDS_GSE.keys()
    #CREEDS_drug_ids_list
    CREEDS_Drug_Final = dict((k, CREEDS_GSE[k]) for k in CREEDS_drug_ids_list)
    CREEDS_drug_final_df = pd.DataFrame(CREEDS_Drug_Final).T
    CREEDS_drug_final_df.columns = ["GSE_ID", "DISEASE"]
    #CREEDS_drug_final_df # DISPLAY THIS DATAFRAME
    ### CREEDS DISEASE CARD FROM DRUG INPUT API
    CREEDS_drug_final_diseases = CREEDS_drug_final_df.DISEASE
    CREEDS_drug_final_GSE_ID = CREEDS_drug_final_df.GSE_ID
    ## CREEDS DISEASE CARD FROM DISEASE QUERY 
    CREEDS_disease_output_ids_up = ["CREEDS_" + s + "_up" for s in CREEDS_drug_final_diseases]
    CREEDS_disease_output_ids_down = ["CREEDS_" + s + "_down" for s in CREEDS_drug_final_diseases]
    loop_iteration = np.arange(0, len(CREEDS_drug_final_diseases))
    loop_iteration = list(loop_iteration)
    CREEDS_total_api_df = []
    CREEDS_all_disease_up_genes = []
    CREEDS_all_disease_down_genes = []
    for a in loop_iteration:
        CREEDS_URL = 'http://amp.pharm.mssm.edu/CREEDS/'
        CREEEDS_Disease_response = requests.get(CREEDS_URL + 'api', params={'id':CREEDS_drug_ids_list[a]})
        if CREEEDS_Disease_response.status_code == 200:
            #pprint(CREEEDS_Disease_response.json())
            #json.dump(CREEEDS_Drug_response.json(), open(CREEDS_drug_final_GSE_ID[a] + '_api1_result.json', 'w'), indent=4)
            CREEEDS_Disease_response_json = CREEEDS_Disease_response.json()
            
            ## up genes
            CREEDS_disease_sigs_up_genes = CREEEDS_Disease_response_json['up_genes']
            CREEDS_disease_sigs_up_genes_df = pd.DataFrame(CREEDS_disease_sigs_up_genes) # this is the up genes dataframe
            CREEDS_disease_sigs_up_genes_df.columns = ["Genes", "Score"]
            #desc = (a + "_" + DrOI + "_" + CREEEDS_Disease_response["geo_id"])
            #CREEDS_desc.append(desc)
            CREEDS_all_disease_up_genes.append(list(CREEDS_disease_sigs_up_genes_df["Genes"]))
            
            filename1 = (str(CREEDS_drug_ids_list[a]) + "_CREEDS_disease_sig_up_genes.csv")
            #CREEDS_disease_sigs_up_genes_df.to_csv(filename1) # this saves the df as a csv
            
            
            ## down genes
            CREEDS_disease_sigs_down_genes = CREEEDS_Disease_response_json['down_genes']
            CREEDS_disease_sigs_down_genes_df = pd.DataFrame(CREEDS_disease_sigs_down_genes) # this is the up genes dataframe
            CREEDS_disease_sigs_down_genes_df.columns = ["Genes", "Score"]
            CREEDS_all_disease_down_genes.append(list(CREEDS_disease_sigs_down_genes_df["Genes"]))
            
            filename2 = (str(CREEDS_drug_ids_list[a]) + "_CREEDS_disease_sig_down_genes.csv")
            #CREEDS_disease_sigs_down_genes_df.to_csv(filename2) # this saves the df as a csv
            print(filename2)
            # entire json
            #json.dump(response.json(), open(a + '_CREEDS_Disease_sig.json', 'w'), indent=4) # if the user wants the entire json, they can download this
                
            
            #CREEEDS_Drug_response_df = pd.DataFrame(CREEEDS_Drug_response_json)
            #CREEEDS_Drug_response_df # This will be the dataframe to return
            #CREEDS_total_api_df.append(CREEEDS_Drug_response_df)
    #CREEDS_total_api_df = pd.concat(CREEDS_total_api_df, axis =1)
    #CREEDS_total_api_df.T ## display this datatable
    ### GENESHOT API and further integration
    GENESHOT_URL = 'http://amp.pharm.mssm.edu/geneshot/api'
    query_string = '/search/%s'
    search_term = DrOI
    # true query from geneshot
    response = requests.get(
        GENESHOT_URL + query_string % (search_term)
    )
    if not response.ok:
        raise Exception('Error during query')
    data = json.loads(response.text)
    #print(data)
    ## GENESHOT QUERY USING AutoRIF
    GENESHOT_URL = 'http://amp.pharm.mssm.edu/geneshot/api'
    query_string = '/search/auto/%s'
    search_term = 'wound healing' # this will be the user input 
    geneshot_response = requests.get(
        GENESHOT_URL + query_string % (search_term)
    )
    if not geneshot_response.ok:
        raise Exception('Error during query')
    geneshot_data = json.loads(geneshot_response.text)
    #print(geneshot_data)
    geneshot_gene_df = geneshot_data["gene_count"]
    geneshot_gene_list = list(geneshot_gene_df.keys()) # this extracts the genes from the json. We can then resend this through the geneshot api
    geneshot_gene_list_commas = ",".join(geneshot_gene_list) # can save this as a csv. 
    geneshot_gene_df1 = pd.DataFrame(geneshot_gene_df).T
    geneshot_gene_df1.columns = ["Pubmed Count", "Publication Count/Total Publications"]
    #write the geneshot pubmed data
    #geneshot_gene_df1.to_csv(search_term + "_geneshot_pubmed_counts.csv")
    query_string = '/associate/%s/%s'
    similarity_matrix = 'coexpression' # we can make this dynamic. Parameters: (generif, tagger, autorif, coexpression, enrichr)
    gene_symbols = geneshot_gene_list_commas
    coexpression_response = requests.get(
        GENESHOT_URL + query_string % (similarity_matrix, gene_symbols)
    )
    if not coexpression_response.ok:
        raise Exception('Error during query')
    coexpression_data = json.loads(coexpression_response.text) # this will be the coexpression json they can download
    geneshot_coexp_ass = {"GENESHOT_coexpression":list(coexpression_data["association"].keys())}
    query_string = '/associate/%s/%s'
    similarity_matrix = 'generif' # we can make this dynamic. Parameters: (generif, tagger, autorif, coexpression, enrichr)
    gene_symbols = geneshot_gene_list_commas
    generif_response = requests.get(
        GENESHOT_URL + query_string % (similarity_matrix, gene_symbols)
    )
    if not generif_response.ok:
        raise Exception('Error during query')
    generif_data = json.loads(generif_response.text) # this will be the coexpression json they can download
    geneshot_generif = {"GENESHOT_generif":list(generif_data["association"].keys())}
    query_string = '/associate/%s/%s'
    similarity_matrix = 'tagger' # we can make this dynamic. Parameters: (generif, tagger, autorif, coexpression, enrichr)
    gene_symbols = geneshot_gene_list_commas
    tagger_response = requests.get(
        GENESHOT_URL + query_string % (similarity_matrix, gene_symbols)
    )
    if not tagger_response.ok:
        raise Exception('Error during query')
    tagger_data = json.loads(tagger_response.text) # this will be the coexpression json they can download
    geneshot_tagger = {"GENESHOT_tagger":list(tagger_data["association"].keys())}
    query_string = '/associate/%s/%s'
    similarity_matrix = 'tagger' # we can make this dynamic. Parameters: (generif, tagger, autorif, coexpression, enrichr)
    gene_symbols = geneshot_gene_list_commas
    autorif_response = requests.get(
        GENESHOT_URL + query_string % (similarity_matrix, gene_symbols)
    )
    if not autorif_response.ok:
        raise Exception('Error during query')
    autorif_data = json.loads(autorif_response.text) # this will be the coexpression json they can download
    geneshot_autorif = {"GENESHOT_autorif":list(autorif_data["association"].keys())}
    query_string = '/associate/%s/%s'
    similarity_matrix = 'tagger' # we can make this dynamic. Parameters: (generif, tagger, autorif, coexpression, enrichr)
    gene_symbols = geneshot_gene_list_commas
    enrichr_response = requests.get(
        GENESHOT_URL + query_string % (similarity_matrix, gene_symbols)
    )
    if not enrichr_response.ok:
        raise Exception('Error during query')
    enrichr_data = json.loads(enrichr_response.text) # this will be the coexpression json they can download
    geneshot_enrichr = {"GENESHOT_enrichr":list(enrichr_data["association"].keys())}
    #### GMT formation from these datasets (NO X2K)
    #### format = TITLE \t\ Description \t\ Genes
    def merge_two_dicts(x, y):
        z = x.copy()   # start with x's keys and values
        z.update(y)    # modifies z with y's keys and values & returns None
        return z
    DrOI_up_no_perts_cp = {"L1000FWD_" + k +"_up": v for k, v in DrOI_up_no_perts.items()}
    DrOI_down_no_perts_cp = {"L1000FWD_" + k +"_down": v for k, v in DrOI_down_no_perts.items()}
    ## Genes
    DrugMatrix_Drug_Genes = merge_two_dicts(DrugMatrix_up_sigs , DrugMatrix_down_sigs)
    DrugMatrix_Drug_Genes = {"DrugMatrix_" + k: v for k, v in DrugMatrix_Drug_Genes.items()}
    L1000_Drug_Genes = merge_two_dicts(DrOI_up_no_perts_cp,DrOI_down_no_perts_cp)
    CREEDS_up_Genes = {
        CREEDS_drug_output_ids_up[a]: CREEDS_all_up_genes[a]
        for a in range(len(CREEDS_drug_output_ids_up))
    }
    CREEDS_down_Genes = {
        CREEDS_drug_output_ids_down[a]: CREEDS_all_down_genes[a]
        for a in range(len(CREEDS_drug_output_ids_up))
    }
    CREEDS_Disease_up_Genes = {
        CREEDS_disease_output_ids_up[a]: CREEDS_all_disease_up_genes[a]
        for a in range(len(CREEDS_drug_output_ids_up))
    }
    CREEDS_Disease_down_Genes = {
        CREEDS_disease_output_ids_down[a]: CREEDS_all_disease_down_genes[a]
        for a in range(len(CREEDS_drug_output_ids_up))
    }
    total_genes = merge_two_dicts(L1000_Drug_Genes, DrugMatrix_Drug_Genes)
    total_genes =merge_two_dicts(total_genes, geneshot_coexp_ass)
    total_genes =merge_two_dicts(total_genes, geneshot_generif)
    total_genes =merge_two_dicts(total_genes, geneshot_tagger)
    total_genes =merge_two_dicts(total_genes, geneshot_autorif)
    total_genes =merge_two_dicts(total_genes, geneshot_enrichr)
    total_genes =merge_two_dicts(total_genes, CREEDS_up_Genes)
    total_genes =merge_two_dicts(total_genes, CREEDS_down_Genes)
    total_genes =merge_two_dicts(total_genes, CREEDS_Disease_up_Genes)
    total_genes =merge_two_dicts(total_genes, CREEDS_Disease_down_Genes)
    ## you will need to change this path file. 
    print("./data/" + DrOI)
    with open ("./data/" + DrOI+".gmt", "w") as file:
        for k in list(total_genes.keys()):
            file.write(k + '\t')
            #file.write('\t'+'na')
            file.write("\t".join(total_genes[k]))
            file.write('\n')

            

#AUTO ENCODER CODE

    

    all_genes = pd.read_csv('./data/AE/genes_info.tsv', sep='\t', index_col=0)
    gmt_fname = "./data/" + DrOI + '.gmt'
    lib_name = os.path.splitext(gmt_fname.rsplit('/', 1)[-1])[0]
    gvm_fname = './data/' + lib_name + '.h5'
    print(gvm_fname)
    formatted_gvm_fname = './data/' + lib_name + '_FORMATTED.h5'
    
    if os.path.isfile(gvm_fname): 
        gvm = open_gvm(gvm_fname)
    else:
        gvm = convert_genesetlist(get_genesetlist(gmt_fname, 'gmt_fname'), to='gvm_h5', output_fname=gvm_fname)
    
    summary = format_gvm_h5(gvm_fname = gvm_fname, all_genes = all_genes,
                       output_fname = formatted_gvm_fname, max_gs_loss=1.0, min_gs_size=1,
                       overwrite = False, return_value='summary')

    n_labels, n_genes = get_gvm_size(formatted_gvm_fname)
    (n_labels, n_genes)

    group = 'AE' # vanilla autoencoder

    batch_size = 128
    m = 1000 # middle dimension
    l = 50 # latent dimension

    model = build_vae(input_dim=n_genes, middle_dim = m, latent_dim = l, 
                    batch_size=batch_size, optimizer='Adamax', lr=.001)
    vae, enc, dec = (model['vae'], model['enc'], model['dec'])
    vae.load_weights('./models/%s/weights/%04dm_%04dl.h5'%(group, m, l))     

    z = enc.predict_generator(
    GeneVec_Generator(formatted_gvm_fname, gvm_path='gvm', batch_size=1000, shuffle=False),
    workers=4, use_multiprocessing=True, verbose=0)

    euc_dist = pairwise_distances(z, metric='euclidean')
    cos_sim = cosine_similarity(z)
    labels = open_gvm(formatted_gvm_fname)['idx']

    euc_dist_df = pd.DataFrame(euc_dist, index=labels, columns=labels)
    cos_sim_df = pd.DataFrame(cos_sim, index=labels, columns=labels) 
    euc_dist_df.iloc[:5, :5]
    cos_sim_df.iloc[:5, :5]
    euc_dist_df.to_pickle('./data/%s_DIST_EUC.pkl'%lib_name)
    cos_sim_df.to_pickle('./data/%s_DIST_COS.pkl'%lib_name)
    cos_sim_df2 = pd.read_pickle('./data/%s_DIST_COS.pkl'%lib_name)

    np.all(cos_sim_df == cos_sim_df2)
# load data into new network instance and cluster
    net = Network()

#net.load_file('cars.tsv')
    net.load_df(cos_sim_df)
    # Z-score normalize the rows
    # net.normalize(axis='row', norm_type='zscore', keep_orig=True)

    # # filter for the top 100 columns based on their absolute value sum
    # net.filter_N_top('col', 100, 'sum')

    net.cluster()
    net.write_json_to_file('viz', 'static/json/mult_view.json')

    K.clear_session()
    return render_template('clustergrammer.html', inputs = input_json, disease_inputs = disease_input_json)

@app.route(ENDPOINT + '/gwas_drugs', methods=['POST'])
def gwas_drugs():
    ### DATA PROCESSING
    data = request.get_json(force=True)


    GWAS_cat = {}

    for line in urllib.request.urlopen('http://amp.pharm.mssm.edu/lincs-playground/index.php/s/9Y78KySXTy4IqOc/download'):
        label, genelist = line.decode().split('\t', maxsplit=1) 
        genelist_split = genelist.strip().split('\t')
        GWAS_cat[label] = genelist_split

    GWAS_phenotypes_list = list(GWAS_cat.keys())
    GWAS_cat = {k.replace('_', ' '): v for k, v in GWAS_cat.items()}
#GWAS_drug_and_disease_list
    GWAS_phenotypes_list_no_spaces = []
    for i in GWAS_phenotypes_list:
        j = i.replace('_',' ')
        GWAS_phenotypes_list_no_spaces.append(j)
#GWAS_drug_and_disease_list

    user_input = data['input'] ## CHANGE THIS TO WHATEVER THE USER INPUT IS (EITHER DISEASE OR DRUGS)

    GWAS_cat_dict ={
        k: v
        for k, v in GWAS_cat.items()
        if re.compile(user_input).search(k)
    }

    GWAS_cat_df = pd.DataFrame(GWAS_cat.items())
    GWAS_cat_df.columns = ["Phenotype", "Genes"]
    GWAS_input_subset = GWAS_cat_df[GWAS_cat_df["Phenotype"].apply(
        lambda s: bool(re.compile(str(user_input), re.IGNORECASE).search(str(s))))] # can display this dataframe (it won't look great. Maybe just display the first column?)
    GWAS_phenotype_df = pd.DataFrame(GWAS_input_subset.Phenotype) #display these names!


    if len(GWAS_cat_dict) > 0:
        GWAS_json = json.dumps(GWAS_cat_dict)
        GWAS_filename = open(user_input + "_GWAS.json", "w") 
        GWAS_filename.write(GWAS_json) # saves the json 
        GWAS_filename.close()
        print("Significant genes for " + user_input + " found in GWAS")
    else:
        print("No significant genes for " + user_input + " found in GWAS")

    return render_template("gwas_drugs.html", gwas_signatures =  GWAS_phenotype_df.to_json(), gwas_json = GWAS_json)

@app.route(ENDPOINT + '/ukbiobank', methods=['POST'])
def biobank():
    data = request.get_json(force=True)


    GWAS_cat = {}

    for line in urllib.request.urlopen('http://amp.pharm.mssm.edu/lincs-playground/index.php/s/9Y78KySXTy4IqOc/download'):
        label, genelist = line.decode().split('\t', maxsplit=1) 
        genelist_split = genelist.strip().split('\t')
        GWAS_cat[label] = genelist_split

    GWAS_phenotypes_list = list(GWAS_cat.keys())
    GWAS_cat = {k.replace('_', ' '): v for k, v in GWAS_cat.items()}
#GWAS_drug_and_disease_list
    GWAS_phenotypes_list_no_spaces = []
    for i in GWAS_phenotypes_list:
        j = i.replace('_',' ')
        GWAS_phenotypes_list_no_spaces.append(j)
#GWAS_drug_and_disease_list

    user_input = data['input'] ## CHANGE THIS TO WHATEVER THE USER INPUT IS (EITHER DISEASE OR DRUGS)

    GWAS_cat_dict ={
        k: v
        for k, v in GWAS_cat.items()
        if re.compile(user_input).search(k)
    }

    GWAS_cat_df = pd.DataFrame(GWAS_cat.items())
    GWAS_cat_df.columns = ["Phenotype", "Genes"]
    GWAS_input_subset = GWAS_cat_df[GWAS_cat_df["Phenotype"].apply(
        lambda s: bool(re.compile(str(user_input), re.IGNORECASE).search(str(s))))] # can display this dataframe (it won't look great. Maybe just display the first column?)
    GWAS_phenotype_df = pd.DataFrame(GWAS_input_subset.Phenotype) #display these names!


    if len(GWAS_cat_dict) > 0:
        GWAS_json = json.dumps(GWAS_cat_dict)
        GWAS_filename = open(user_input + "_GWAS.json", "w") 
        GWAS_filename.write(GWAS_json) # saves the json 
        GWAS_filename.close()
        print("Significant genes for " + user_input + " found in GWAS")
    else:
        print("No significant genes for " + user_input + " found in GWAS")

    return render_template("ukbiobank.html", gwas_signatures =  GWAS_phenotype_df.to_json(), gwas_json = GWAS_json)
