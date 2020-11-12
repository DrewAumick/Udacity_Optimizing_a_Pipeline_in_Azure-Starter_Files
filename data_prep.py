import numpy as np
import pandas as pd
from azureml.core.workspace import Workspace
from azureml.core import Dataset
from azureml.core.datastore import Datastore
from azureml.data.dataset_factory import TabularDatasetFactory
from sklearn.preprocessing import LabelEncoder

# Searches the workspace for a dataset of the given key. If it doesn't find it, it creates the dataset from a CSV file
def get_DDoS_dataset(ws, key="Clean DDoS Dataset"):   
    if key in ws.datasets.keys(): 
            dataset = ws.datasets[key] 
    else:
            create_DDoS_datasets(ws)
            dataset = ws.datasets[key] 
            
    return dataset

# Download data from kaggle, create the DDoS Dataset, clean it, and register both the base version and the cleaned version to Workspace
def create_DDoS_datasets(ws):  
    dtypes = {
        'Src IP': 'category',
        'Src Port': 'uint16',
        'Dst IP': 'category',
        'Dst Port': 'uint16',
        'Protocol': 'category',
        'Flow Duration': 'uint32',
        'Tot Fwd Pkts': 'uint32',
        'Tot Bwd Pkts': 'uint32',
        'TotLen Fwd Pkts': 'float32',
        'TotLen Bwd Pkts': 'float32',
        'Fwd Pkt Len Max': 'float32',
        'Fwd Pkt Len Min': 'float32',
        'Fwd Pkt Len Mean': 'float32',
        'Fwd Pkt Len Std': 'float32',
        'Bwd Pkt Len Max': 'float32',
        'Bwd Pkt Len Min': 'float32',
        'Bwd Pkt Len Mean': 'float32',
        'Bwd Pkt Len Std': 'float32',
        'Flow Byts/s': 'float32',
        'Flow Pkts/s': 'float32',
        'Flow IAT Mean': 'float32',
        'Flow IAT Std': 'float32',
        'Flow IAT Max': 'float32',
        'Flow IAT Min': 'float32',
        'Fwd IAT Tot': 'float32',
        'Fwd IAT Mean': 'float32',
        'Fwd IAT Std': 'float32',
        'Fwd IAT Max': 'float32',
        'Fwd IAT Min': 'float32',
        'Bwd IAT Tot': 'float32',
        'Bwd IAT Mean': 'float32',
        'Bwd IAT Std': 'float32',
        'Bwd IAT Max': 'float32',
        'Bwd IAT Min': 'float32',
        'Fwd PSH Flags': 'category',
        'Bwd PSH Flags': 'category',
        'Fwd URG Flags': 'category',
        'Bwd URG Flags': 'category',
        'Fwd Header Len': 'uint32',
        'Bwd Header Len': 'uint32',
        'Fwd Pkts/s': 'float32',
        'Bwd Pkts/s': 'float32',
        'Pkt Len Min': 'float32',
        'Pkt Len Max': 'float32',
        'Pkt Len Mean': 'float32',
        'Pkt Len Std': 'float32',
        'Pkt Len Var': 'float32',
        'FIN Flag Cnt': 'category',
        'SYN Flag Cnt': 'category',
        'RST Flag Cnt': 'category',
        'PSH Flag Cnt': 'category',
        'ACK Flag Cnt': 'category',
        'URG Flag Cnt': 'category',
        'CWE Flag Count': 'category',
        'ECE Flag Cnt': 'category',
        'Down/Up Ratio': 'float32',
        'Pkt Size Avg': 'float32',
        'Fwd Seg Size Avg': 'float32',
        'Bwd Seg Size Avg': 'float32',
        'Fwd Byts/b Avg': 'uint32',
        'Fwd Pkts/b Avg': 'uint32',
        'Fwd Blk Rate Avg': 'uint32',
        'Bwd Byts/b Avg': 'uint32',
        'Bwd Pkts/b Avg': 'uint32',
        'Bwd Blk Rate Avg': 'uint32',
        'Subflow Fwd Pkts': 'uint32',
        'Subflow Fwd Byts': 'uint32',
        'Subflow Bwd Pkts': 'uint32',
        'Subflow Bwd Byts': 'uint32',
        'Init Fwd Win Byts': 'uint32',
        'Init Bwd Win Byts': 'uint32',
        'Fwd Act Data Pkts': 'uint32',
        'Fwd Seg Size Min': 'uint32',
        'Active Mean': 'float32',
        'Active Std': 'float32',
        'Active Max': 'float32',
        'Active Min': 'float32',
        'Idle Mean': 'float32',
        'Idle Std': 'float32',
        'Idle Max': 'float32',
        'Idle Min': 'float32',
        'Label': 'category'
    }

    data = pd.read_csv(
            './final_dataset.csv',
            parse_dates=['Timestamp'],
            usecols=[*dtypes.keys(), 'Timestamp'],
            engine='c',
            low_memory=True,
            na_values=np.inf
        )

    # There are over 12 million rows in this orignal dataset. For this project, that much data is taking far too long, so I'm randomly sampling only .5% of the data
    data = data.sample(frac=0.005)

    # Register Base Dataset in Workspace
    datastore = Datastore(ws)
    name = "DDoS Dataset"
    description_text = "DDoS DataSet for Udacity Capstone Project"
    dataset = TabularDatasetFactory.register_pandas_dataframe(data,
                               datastore,
                               name,
                               description=description_text)
    
    # Clean dataset and register the clean version
    cleaned_data = clean_data(data)
    
    clean_dataset_name = "Clean DDoS Dataset"
    clean_description_text = description_text + " that has been cleaned"
    clean_dataset = TabularDatasetFactory.register_pandas_dataframe(cleaned_data,
                               datastore,
                               clean_dataset_name,
                               description=clean_description_text)
    
# Clean the data    
def clean_data(df):
    LE = LabelEncoder()
    df['Src IP'] = LE.fit_transform(df['Src IP'])    
    df['Dst IP'] = LE.fit_transform(df['Dst IP'])
    
    df['Timestamp'] = df['Timestamp'].apply(lambda s: s.value)    
        
    # Drop columns that don't impact classification 
    colsToDrop = np.array(['Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg'])
    
    # Drop columns where missing values are more than 50% 
    missing = df.isna().sum()
    missing = pd.DataFrame({'count': missing, '% of total': missing/len(df)*100}, index=df.columns)
    colsToDrop = np.union1d(colsToDrop, missing[missing['% of total'] >= 50].index.values)

    # Drop rows where a column missing values are no more than 5%
    dropnaCols = missing[(missing['% of total'] > 0) & (missing['% of total'] <= 5)].index.values
    df = df['Flow Byts/s'].replace(np.inf, np.nan)
    df = df['Flow Pkts/s'].replace(np.inf, np.nan)
    
    dropnaCols = np.union1d(dropnaCols, ['Flow Byts/s', 'Flow Pkts/s'])
    df.dropna(subset=dropnaCols, inplace=True)

    df.drop(columns=colsToDrop, inplace=True)
    
    # convert the label to 1 for ddos or 0 for benign
    df["Label"] = df.Label.apply(lambda s: 1 if s == "ddos" else 0)  

    df = df.fillna(0)  
        
    return df

def main():
    ws = Workspace.from_config()
    
    df = get_DDoS_dataset(ws)

if __name__ == '__main__':
    main()