import argparse
import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

import joblib
from azureml.core import Run, Dataset
from azureml.core.workspace import Workspace
from azureml.core.authentication import MsiAuthentication

#from data_prep import get_DDoS_dataset
run = Run.get_context() 

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=100, help="Number of trees in the forest")
    parser.add_argument('--max_depth', type=int, default=None, help="The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.")
    parser.add_argument('--min_samples_split', type=int, default=2, help="The minimum number of samples required to split an internal node.")
    parser.add_argument('--min_samples_leaf', type=int, default=1, help="The minimum number of samples required to be at a leaf node.")
    
    args = parser.parse_args()
    if args.max_depth == 0:
        max_depth = None
    else:
        max_depth = args.max_depth

    run.log("Num Estimators:", np.float(args.n_estimators))
    run.log("Max Depth:", max_depth)
    run.log("Min Samples Split:", np.int(args.min_samples_split))
    run.log("Min Samples Leaf:", np.int(args.min_samples_leaf))

    workspace = run.experiment.workspace
    dataset_name = 'Clean DDoS Dataset'
    dataset = Dataset.get_by_name(workspace=workspace, name=dataset_name)

    df = dataset.to_pandas_dataframe()

    y = df.pop("Label")

    x_train, x_test, y_train, y_test = train_test_split(df,y)

    model = RandomForestClassifier(n_estimators=args.n_estimators, 
                                   max_depth=max_depth, 
                                   min_samples_split=args.min_samples_split, 
                                   min_samples_leaf=args.min_samples_leaf, 
                                   )
    
    model = model.fit(x_train,y_train)

    joblib.dump(model, './outputs/model.joblib')

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()
