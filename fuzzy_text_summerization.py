import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd
import numpy as np
import os


df1 = pd.read_csv('DATA/tfisf_unique.csv')
df1 =sorted(list(np.array(df1.iloc[:, 1])))
df2 = pd.read_csv('DATA/capital_unique.csv')
df2 = sorted(list(np.array(df2.iloc[:, 1])))
df3 = pd.read_csv('DATA/sent_similarity_unique.csv')
df3 = sorted(list(np.array(df3.iloc[:, 1])))
df4 = pd.read_csv('DATA/keyword_unique.csv')
df4 = sorted(list(np.array(df4.iloc[:, 1])))
df5 = pd.read_csv('DATA/content_words_unique.csv')
df5 = sorted(list(np.array(df5.iloc[:, 1])))

#input variables for the fuzzy control system
tfisf_var = ctrl.Antecedent(df1, 'tfisf_var')
cap_var = ctrl.Antecedent(df2, 'cap_var')
sim_var = ctrl.Antecedent(df3, 'sim_var')
kw_var = ctrl.Antecedent(df4, 'kw_var')
cont_var = ctrl.Antecedent(df5, 'cont_var')

#ouput variable for the fuzzy control system
senten = ctrl.Consequent(np.arange(0, 100), 'senten')

print("variables initiated")


train_feat_path = 'DATA/training_data_feat_complete.csv'
train_file = pd.read_csv(train_feat_path)
param = {}
for col in list(train_file.columns)[2:-1]:
  col_values = train_file[col]
  mean = np.mean(col_values)
  stdev = np.std(col_values)

  sorted_col_values = sorted(col_values)
  for i in range(len(sorted_col_values)):
    if sorted_col_values[i] == mean:
      start = i
      for j in range(i, len(sorted_col_values)):
        if sorted_col_values[j] > mean:
          break
      end = j
      break
    elif sorted_col_values[i] > mean:
      start = i
      end = i
      break
  L_M = np.sum(sorted_col_values[:end])
  R_M = np.sum(sorted_col_values[start:])
  S = L_M/float(R_M)

  sd_R = stdev/(1+S)
  sd_L = sd_R*S

  LL = mean - sd_L
  UL = mean + sd_R

  param[col] = [LL, mean, UL]

print("membership functions initiated")

tfisf_var['low'] = fuzz.trimf(tfisf_var.universe, [0, 0, param['tfisf'][1]])
tfisf_var['mid'] = fuzz.trimf(tfisf_var.universe, [param['tfisf'][0], param['tfisf'][1], param['tfisf'][2]])
tfisf_var['high'] = fuzz.trimf(tfisf_var.universe, [param['tfisf'][1], 9.962982124321767, 9.962982124321767])

cap_var['low'] = fuzz.trimf(cap_var.universe, [0, 0, param['capital'][1]])
cap_var['mid'] = fuzz.trimf(cap_var.universe, [param['capital'][0], param['capital'][1], param['capital'][2]])
cap_var['high'] = fuzz.trimf(cap_var.universe, [param['capital'][1], 1.0, 1.0])

sim_var['low'] = fuzz.trimf(sim_var.universe, [0, 0, param['sent_similarity'][1]])
sim_var['mid'] = fuzz.trimf(sim_var.universe, [param['sent_similarity'][0], param['sent_similarity'][1], param['sent_similarity'][2]])
sim_var['high'] = fuzz.trimf(sim_var.universe, [param['sent_similarity'][1], 1.0, 1.0])

kw_var['low'] = fuzz.trimf(kw_var.universe, [0, 0, param['keyword'][1]])
kw_var['mid'] = fuzz.trimf(kw_var.universe, [param['keyword'][0], param['keyword'][1], param['keyword'][2]])
kw_var['high'] = fuzz.trimf(kw_var.universe, [param['keyword'][1], 1.0, 1.0])

cont_var['low'] = fuzz.trimf(cont_var.universe, [0, 0, param['content_words'][1]])
cont_var['mid'] = fuzz.trimf(cont_var.universe, [param['content_words'][0], param['content_words'][1], param['content_words'][2]])
cont_var['high'] = fuzz.trimf(cont_var.universe, [param['content_words'][1], 1.0, 1.0])

senten['very bad'] = fuzz.trimf(senten.universe, [0, 0, 25])
senten['bad'] = fuzz.trimf(senten.universe, [0, 25, 50])
senten['avg'] = fuzz.trimf(senten.universe, [25, 50, 75])
senten['good'] = fuzz.trimf(senten.universe, [50, 75, 100])
senten['best'] = fuzz.trimf(senten.universe, [75, 100, 100])

rule1 = ctrl.Rule(sim_var['low'], senten['very bad'])
rule2 = ctrl.Rule(sim_var['mid'] & (cont_var['high'] | cont_var['mid']), senten['bad'])
rule3 = ctrl.Rule(sim_var['high'] & (cont_var['high'] | cont_var['mid']), senten['avg'])
rule4 = ctrl.Rule(sim_var['mid'] & cont_var['low'], senten['avg'])
rule5 = ctrl.Rule(sim_var['high'] & cont_var['low'], senten['good'])
rule6 = ctrl.Rule(tfisf_var['high'], senten['best'])
rule7 = ctrl.Rule(tfisf_var['mid'] & cont_var['low'], senten['good'])
rule8 = ctrl.Rule(tfisf_var['low'], senten['very bad'])
rule9 = ctrl.Rule(tfisf_var['mid'] & (cont_var['high'] | cont_var['mid']), senten['avg'])

sent_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
Sent = ctrl.ControlSystemSimulation(sent_ctrl)

print("FLS created")

def find_summary_sentences(scores, n):
  import copy
  temp = sorted(list(set(copy.copy(scores))), reverse=True)
  if n < len(temp):
    temp = temp[:n]
  temp2 = []
  for x in temp:
    temp2.append((x[1], x[0]))
  return len(temp2), sorted(temp2)

def find_summary(file, n):
  summary=[]
  scores1 = []
  scores2 = []
  for i in range(len(file)):
    Sent.input['tfisf_var'] = file.iloc[i, 2]
    Sent.input['sim_var'] = file.iloc[i, 4]
    Sent.input['cont_var'] = file.iloc[i, 6]
    Sent.compute()
    x = Sent.output['senten']
    #Sent.print_state()
    if x >= 75:
      scores1.append((x, i))
    elif x >= 50 and x < 75:
      scores2.append((x, i))
  n1, first = find_summary_sentences(scores1, n)
  if n1 != n:
    n2, second = find_summary_sentences(scores2, n-n1)
    first.extend(second)
  index = sorted(first)
  for x in index:
      summary.append(file.iloc[x[0], 1])
  for i in range(len(summary)):
    x = summary[i]
    try:
      x = x.replace('\n', ' ')
    except:
      x = x
    summary[i] = x
  return summary


path = 'DATA/validation_features'
cont = os.listdir(path)

os.mkdir('system')
i = 0
for x in cont:
 try:
  pt_fl = os.path.join(path, x)
  df = pd.read_csv(pt_fl)
  lines = find_summary(df, 50)
  name = x.split('.')[0]+'_system1.txt'
  fl = open('system/'+name, 'w')
  for line in lines:
    fl.write(line+'\n')
  fl.close()
  print(len(cont)-i)
  i += 1
 except:
  print("ERROR: "+x)
