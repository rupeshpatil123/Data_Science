import pandas as pd
import pickle
df = pd.DataFrame([]) #creating a blank dataframe


### data extraction
"""
    a function that accepts a path 
    from where the data is to be loaded
    and stores the reference of this data frame
    into a non-local variable
"""

def source_data(path : str) -> None:
    global df
    temp_df = pd.read_csv(path)
    df = temp_df.copy()
    del temp_df


"""
    data exploration
"""

def data_explore() -> None:
    f1 = open("log file", "w")
    print(f"Shape:\n {df.shape}", file=f1)
    print(f"Index:\n {df.index}", file=f1)
    print(f"Columns:\n {df.columns}", file=f1)
    print(f"Unique values:\n {df.nunique()}", file=f1)
    print(f"Null count per column\n{df.isna().sum()}", file=f1)


def data_cleaning()->None:
    global df

    condition = (df["age"] >= 0) #defining your filter

    #apply filter now!
    df = df[    condition    ].copy() #makes new table "df"

    df.dropna(inplace=True) #drop missing values. DO THIS INPLACE meaning in the same object!

    df.reset_index(drop=True, inplace=True)

def preserve_model()->None:
    pickle.dump( df, open("clean_data.pkl", "wb")  )


