import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import re
import json
import os
from libchebipy._chebi_entity import ChebiEntity
import urllib.parse

UMLS_API_KEY = 'your_umls_api_key'  # Replace

def find_definition_from_collections(uri, file_paths: list[str]):
    definitions = set()
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)
        for pmid, content in data.items():
            if 'entities' in content:
                for entity in content['entities']:
                    if entity['uri'] == uri:
                        definitions.add(entity['text_span'])

    return list(definitions)

def find_definition_chebi(uri, download_path='chebi_cache', get_synonyms=True):
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    os.environ['LIBCHEBIPY_DOWNLOAD_DIR'] = download_path
    chebi_entity = ChebiEntity(uri.split('_')[-1])
    definitions = []
    tmp_definitions = chebi_entity.get_names()
    if get_synonyms and len(tmp_definitions) > 0:
        for entry in tmp_definitions:
            if entry._Name__name:
                definitions.append(entry._Name__name)
    else:
        definitions = [chebi_entity.get_name()]
    return definitions

def find_definition_omit(uri):
    """Extract definition from OMIT ontology XML"""
    try:
        response = requests.get(uri)
        # Parse the XML response
        root = ET.fromstring(response.content)
        
        # Define namespaces
        namespaces = {
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
            'owl': 'http://www.w3.org/2002/07/owl#'
        }
        
        # Find the Class element with the matching rdf:about attribute
        class_element = root.find(f'.//owl:Class[@rdf:about="{uri}"]', namespaces)
        
        if class_element is not None:
            # Find the rdfs:label within this class
            label_element = class_element.find('rdfs:label', namespaces)
            if label_element is not None and label_element.text:
                return [label_element.text]
        
        return []
    except Exception as e:
        print(f"Error processing OMIT URI {uri}: {e}")
        return []
    
def find_definition_stato(uri):
    def double_url_encode(url: str) -> str:
        # First URL encode
        once_encoded = urllib.parse.quote(url, safe='')
        # Second URL encode the already encoded string
        twice_encoded = urllib.parse.quote(once_encoded, safe='')
        return twice_encoded
    
    try:
        # Fix the query parameter (was ?lang?en, should be ?lang=en)
        response = requests.get(f"https://www.ebi.ac.uk/ols4/api/v2/ontologies/stato/classes/{double_url_encode(uri)}")
        
        if response.status_code != 200:
            print(f"Error: HTTP {response.status_code} for URI {uri}")
            return []
            
        data = response.json()
        return data['label'] if 'label' in data else []
        
    except Exception as e:
        print(f"Error processing STATO URI {uri}: {e}")
        return []
    

def find_definition_ontobee(uri):
    if uri.find('CHEBI') == -1:
        response = requests.get(f'http://purl.obolibrary.org/obo/{uri.split("/")[-1]}')
    else:
        #response = requests.get(f'https://ontobee.org/ontology/CHEBI?iri=http://purl.obolibrary.org/obo/{uri.split("/")[-1]}')
        return find_definition_chebi(uri, get_synonyms=True)
    # Parse the XML response
    root = ET.fromstring(response.content)
    # Define the namespace for oboInOwl
    namespaces = {
        'oboInOwl': 'http://www.geneontology.org/formats/oboInOwl#'
    }
    # Find all oboInOwl:hasExactSynonym elements
    exact_synonyms = root.findall('.//oboInOwl:hasExactSynonym', namespaces)
    # Extract the text values
    definitions = []
    for synonym in exact_synonyms:
        if synonym.text:
            definitions.append(synonym.text)
    return definitions

def find_definition_umls(uri):
    response = requests.get(f'https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{uri.split("/")[-1]}/atoms?apiKey={UMLS_API_KEY}')
    response = response.json()
    definitions = []
    for v in response['result']:
        if v['language'] == 'ENG':
            definitions.append(v['name'])
    if len(definitions) == 0:
        # No English definition found in atoms, try definitions endpoint
        response = requests.get(f'https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{uri.split("/")[-1]}?apiKey={UMLS_API_KEY}')
        response = response.json()
        definitions = []
        if 'result' in response:
            if 'name' in response['result']:
                definitions.append(response['result']['name'])
    return definitions

def find_definition_mesh(uri):
    # Make the request to MeSH
    response = requests.get(f"https://www.ncbi.nlm.nih.gov/mesh/{uri.split('/')[-1]}")
    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    # Extract the title
    title_tag = soup.find('title')
    if title_tag:
        title_text = title_tag.get_text()
        # Extract the main term from title (remove " - MeSH - NCBI" part)
        main_term = title_text.replace(' - MeSH - NCBI', '')
        #print(f"Main term from title: {main_term}")
    else:
        #print("Title not found")
        pass
    # Extract entry terms
    # Look for the "Entry Terms:" section
    entry_terms = []
    entry_terms_section = soup.find(string="Entry Terms:")
    if entry_terms_section:
        # Find the parent element and then the ul list that follows
        parent = entry_terms_section.parent
        ul_list = parent.find_next('ul')
        if ul_list:
            # Extract all li elements
            li_elements = ul_list.find_all('li')
            for li in li_elements:
                entry_terms.append(li.get_text().strip())
            
            #print(f"\nFound {len(entry_terms)} entry terms:")
            #for i, term in enumerate(entry_terms, 1):
            #    print(f"{i}. {term}")
        else:
            #print("Entry terms list not found")
            pass
    else:
        #print("Entry Terms section not found")
        pass

    # Combine all terms (main term + entry terms)
    definitions = [main_term] + entry_terms
    return definitions

def find_definition_custom_ontology(uri):
    parts = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', uri.split('/')[-1])
    definition = ' '.join(parts)
    return [definition]

def find_definition(uri):
    if uri.find('umls') != -1:
        # UMLS entity
        return find_definition_umls(uri)
    elif uri.find('OMIT_') != -1 or uri.find('NCBITaxon_') != -1 or uri.find('OHMI_') != -1 or uri.find('OGMS_') != -1:
        # OMIT ontology entity
        return find_definition_omit(uri)
    elif uri.find('STATO_') != -1:
        # STATO ontology entity
        return find_definition_stato(uri)
    elif uri.find('CHEBI') != -1:
        return find_definition_chebi(uri, get_synonyms=True)
    elif uri.find('purl') != -1:
        # Ontobee entity
        return find_definition_ontobee(uri)
    elif uri.find('mesh') != -1:
        # MeSH entity
        return find_definition_mesh(uri)
    elif uri.find('w3id') != -1:
        # Custom entity
        return find_definition_custom_ontology(uri)
    else:
        print(f'URI {uri} is not recognized.')
        return []