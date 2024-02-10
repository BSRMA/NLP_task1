import subprocess
import os 

os.mkdir("DATA")
subprocess.call(["python3", "training_feature_extraction.py"])

print("training features extracted")

subprocess.call(["python3", "validation_feature_extraction.py"])

print("validation features extracted")

subprocess.call(["python3", "complete_training_feature_extraction.py"])

print("complete training features merged")

subprocess.call(["python3", "unique_features.py"])

print("unique features found")

subprocess.call(["python3", "fuzzy_text_summerization.py"])

print("\ncreating the test-summerization folder for rouge 2.0")

subprocess.call(["python3", "rename.py"])

print("Created. You may calculate rougue score using this folder.")

os.mkdir('RESULTS')
os.mknod('RESULTS/results.csv')
subprocess.call(["java", "-jar", "rouge2-1.2.2.jar"])

subprocess.call(["python3", "calculate_average_rouge.py"])

print("results in RESULTS folder.")
