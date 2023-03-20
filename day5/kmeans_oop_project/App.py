from Operations import Operations


if __name__ == "__main__":
    obj = Operations(r"C:\Users\harsh\Desktop\NPCI-Python-ML\day5\kmeans_oop_project\defaulters_data.csv")
    obj.explore_data()
    obj.clean_data()
    obj.create_model()
