import pandas as pd
import numpy as np

def calculate_avg_rouge(path):
  df = pd.read_csv(path)
  rougue_type = sorted(list(set(list(df['ROUGE-Type']))))
  task_name = list(set(list(df['Task Name'])))
  metrics = ['Avg_Recall',	'Avg_Precision',	'Avg_F-Score']

  params = {}
  for rt in rougue_type:
    params[rt] = []

  for i in range(len(df)):
    x = df.iloc[i, :]
    if x[0] in rougue_type:
      params[x[0]].append(x)

  for rt in rougue_type:
    params[rt] = pd.DataFrame(params[rt])

  avg_recall = []
  avg_precision = []
  avg_F_score = []
  metric = []
  for rt in rougue_type:
    data = params[rt]
    metric.append(rt)
    avg_recall.append(np.mean(data['Avg_Recall']))
    avg_precision.append(np.mean(data['Avg_Precision']))
    avg_F_score.append(np.mean(data['Avg_F-Score']))

    avg_results = {"metric": metric,
              "Avg_Recall": avg_recall,
               "Avg_Precision": avg_precision,
               "Avg_F-Score": avg_F_score}

    df2 = pd.DataFrame(avg_results)
    df2.to_csv('RESULTS/average_results.csv')

calculate_avg_rouge('RESULTS/results.csv')
