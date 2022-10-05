import pandas as pd
import numpy as np


df=pd.read_csv("Code_python/ffhq_aging_labels.csv")

print("Description of dataset\n", df.describe())
print("First 6 lines of dataset\n", df.head(6))
print("Null elements in dataset\n",df.isnull().sum())

print("Frequency in gender_confidence\n")
print(df.groupby(pd.cut(df["gender_confidence"], np.arange(0, 1.1, 0.1))).count().image_number)
print("\n")


print("Frequency in age_group_confidence\n")
print((df.groupby(pd.cut(df["age_group_confidence"], np.arange(0, 1.1, 0.1))).count()).image_number)
print("\n")


print("Frequency in gender\n")
print(df.groupby((df["gender"])).count().image_number)
print("\n")


print("Frequency in age_group")
print(df.groupby((df["age_group"])).count().image_number)
print("\n")

print("Frequency in group w.r.t. gender")
print(df.groupby((["glasses","gender"])).size())
print("\n")


