
import pandas as pd

from sklearn.metrics import silhouette_score

from sklearn.cluster import KMeans

from pre_model import source_data, data_explore, data_cleaning,preserve_model, df 

from model_creation import create_model,load_data_frame


if __name__ == "__main__":

    source_data(path=r"C:\Users\harsh\Desktop\NPCI-Python-ML\day5\kmeans_defaulters_project\defaulters_data.csv")

    data_explore()

    data_cleaning()

    preserve_model()

    load_data_frame(r"C:\Users\harsh\Desktop\NPCI-Python-ML\day5\kmeans_defaulters_project\clean_data.pkl")
    
    create_model()


