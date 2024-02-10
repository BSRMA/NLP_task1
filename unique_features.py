import pandas as pd
import os

annual_report_path = 'DATA/training_data_feat_complete.csv'

df = pd.read_csv(annual_report_path)
columns = ['tfisf', 'capital', 'sent_similarity', 'keyword', 'content_words', 'class']

dest = 'DATA'
for col in columns:
  x = list(set(df[col]))
  df2 = pd.DataFrame(x)
  name = col+'_unique.csv'
  path = os.path.join(dest, name)
  
  df2.to_csv(path)

