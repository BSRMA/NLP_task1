import pandas as pd
import os

annual_report_path = 'DATA/training_features'
contents1 = os.listdir(annual_report_path)
j = 0


columns = ['Sentences', 'tfisf', 'capital', 'sent_similarity', 'keyword', 'content_words', 'class']
params = {}
for col in columns:
  params[col] = []

for x in contents1:  
  path = os.path.join(annual_report_path, x)
  df = pd.read_csv(path)

  for col in columns:
    y = list(df[col])
    params[col].extend(y)
  
df = pd.DataFrame(params)
df.to_csv('DATA/training_data_feat_complete.csv')
