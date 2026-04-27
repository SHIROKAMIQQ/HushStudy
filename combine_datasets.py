import pandas as pd
import glob

csv_folder = glob.glob("datasets/*.csv")
df_list = [pd.read_csv(file) for file in csv_folder]
master_df = pd.concat(df_list, ignore_index=True)
master_df.to_csv("datasets/master.csv", index=False)
print("Combined all .csv files in datasets/ into datasets/master.csv")
