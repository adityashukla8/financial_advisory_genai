from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import os
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.ai.documentintelligence.models import AnalyzeResult
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
import pandas as pd
import io
from ipdb import set_trace as ipdb
from dotenv import load_dotenv
load_dotenv()

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_name = "bankstatements"
document_analysis_client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

def analyze_document_from_blob(blob_data):
    poller = document_analysis_client.begin_analyze_document("prebuilt-document", blob_data)
    result = poller.result()
    
    return result

blob_container_client = blob_service_client.get_container_client(container_name)
blobs_list = blob_container_client.list_blobs()

def save_tables_to_csv(result, blob_name):
    blob_name = blob_name.replace('customer_id_88/', '').replace('.pdf', '.csv')

    tables = []
    table_dict = {}

    for table in result.tables:
        for cell in table.cells:
            if cell.row_index not in table_dict:
                table_dict[cell.row_index] = {}
            table_dict[cell.row_index][cell.column_index] = cell.content

    if table_dict:
        df = pd.DataFrame.from_dict(table_dict, orient='index').sort_index().sort_index(axis=1)
        csv_data = df.to_csv(index=False)
   
    blob_client = blob_container_client.get_blob_client(f'customer_id_88_csvs/{blob_name}')
    blob_client.upload_blob(csv_data, overwrite=True)


for blob in blobs_list:
    print(blob.name)
    blob_data = blob_service_client.get_blob_client(blob=blob.name, container=container_name).download_blob().readall()
    result = analyze_document_from_blob(blob_data)
    save_tables_to_csv(result, blob.name)

