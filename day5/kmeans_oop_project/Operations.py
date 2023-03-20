import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# self as the first argument in any function indicates that this function should only be called by an instance of the class

class Operations:

    #ACTS AS constructor of the class
    def __init__(self, data_path : str) -> None:
        #initialize object by passing path
        self._data_path = data_path 
        self._data_frame = pd.read_csv(data_path)


    def explore_data(self) ->None:
        with open("log file.txt", "w") as f1:
            print(f"Shape:\n {self._data_frame.shape}", file=f1)
            print(f"Index:\n {self._data_frame.index}", file=f1)
            print(f"Columns:\n {self._data_frame.columns}", file=f1)
            print(f"Unique values:\n {self._data_frame.nunique()}", file=f1)
            print(f"Null count per column\n{self._data_frame.isna().sum()}", file=f1)


    def clean_data(self) -> None:
        condition = (self._data_frame["age"] >= 0) #defining your filter
        #apply filter now!
        self._data_frame = self._data_frame[    condition    ].copy() #makes new table "self._data_frame"
        self._data_frame.dropna(inplace=True) #drop missing values. DO THIS INPLACE meaning in the same object!
        self._data_frame.reset_index(drop=True, inplace=True)

    def create_model(self) -> None:
        features  = ["income", "age"]
        model = KMeans(n_clusters=4, n_init="auto")

        values= model.fit_predict(self._data_frame[features])
        predictions = pd.DataFrame(values, columns=["predicted_cluster"])
        result_df = pd.concat(   [self._data_frame, predictions], axis=1    )
        result_df["predicted_cluster"] = result_df["predicted_cluster"].astype("object")

        ans = silhouette_score(result_df[features],result_df["predicted_cluster"] )
        print(f"Silhouette score: {ans}" )
