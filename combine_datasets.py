import pandas as pd
import glob

"""
We have two folders of data: chatter_classifier_datasets and duration_prediction_datasets.
We want to combine the datasets in each folder such that each folder will have a master.csv file.
This master.csv file will be the one used by the models.
"""

csv_folder = glob.glob("chatter_classifier_datasets/*.csv")
df_list = [pd.read_csv(file) for file in csv_folder]
master_df = pd.concat(df_list, ignore_index=True)
master_df.to_csv("chatter_classifier_datasets/master.csv", index=False)
print("Combined all .csv files in chatter_classifier_datasets/ into chatter_classifier_datasets/master.csv")


csv_folder = glob.glob("duration_prediction_datasets/*.csv")
df_list = [pd.read_csv(file) for file in csv_folder]
master_df = pd.concat(df_list, ignore_index=True)
master_df.to_csv("duration_prediction_datasets/master.csv", index=False)
print("Combined all .csv files in duration_prediction_datasets/ into duration_prediction_datasets/master.csv")
