import pandas as pd

# df = pd.read_parquet("hf://datasets/jlh/home-credit/data/train-00000-of-00001-e68d01965482ae18.parquet")
# df.to_parquet(r"C:\Users\Danch\PycharmProjects\Credit_Scoring_Model_w_transformers\data\home_credit_.parquet")

splits = {'train': 'train.csv', 'test': 'test.csv'}
df = pd.read_csv("hf://datasets/farizcolan/creditscoring/" + splits["train"])

df.to_csv(r"C:\Users\Danch\PycharmProjects\Credit_Scoring_Model_w_transformers\data\credit_scoring.csv")