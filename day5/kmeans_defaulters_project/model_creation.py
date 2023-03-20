import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import pickle

df = pd.DataFrame([])

"""
import the model from pickle file
"""

def load_data_frame(path : str)->None:
    global df
    f1 = open(path, "rb")
    
    temp_df = pickle.load(f1)
    df = temp_df.copy()

    del temp_df
    f1.close()

"""
create model. predict clusters. append everything to a result table
for future reference
"""

def create_model():
    global df
    features  = ["income", "age"]
    model = KMeans(n_clusters=4, n_init="auto")

    values= model.fit_predict(df[features])

    predictions = pd.DataFrame(values, columns=["predicted_cluster"])


    result_df = pd.concat(   [df, predictions], axis=1    )

    result_df["predicted_cluster"] = result_df["predicted_cluster"].astype("object")


    ans = silhouette_score(result_df[features],result_df["predicted_cluster"] )
    print(f"Silhouette score: {ans}" )