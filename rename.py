import shutil
import os

src1 = 'system'
src2 = 'validation/gold_summaries'
  
os.mkdir('projects')
os.mkdir('projects/test-summarization')

dest = 'projects/test-summarization'
shutil.copy(src1, dest) 

os.mkdir('reference')
ref = 'reference'
contents = os.listdir(src2)
for x in contents:
  path1 = os.path.join(src2, x)
  temp = x.split('_')
  y = str(temp[0])+'_reference'
  temp2 = temp[1].split('.')
  y += str(temp2[0])
  y += '.txt'
  path2 = os.path.join(ref, y)
  shutil.copy(path1, path2) 
  
shutil.move('reference', dest) 
