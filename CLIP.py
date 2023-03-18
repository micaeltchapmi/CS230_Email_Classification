import os
import torch
from transformers import pipeline
import glob

data_dir = "./data/EC_Data/test"
model_name = "openai/clip-vit-large-patch14-336"
classifier = pipeline("zero-shot-image-classification", model = model_name, device=0)
labels_for_classification =  os.listdir(data_dir)

img_paths = glob.glob(data_dir + "/*/*")
N = float(len(img_paths))

top_1_count = 0
top_3_count = 0
top_5_count = 0
top_10_count = 0

for i, img_path in enumerate(img_paths):
  print("processing %d / %d" % (i, N))
  correct_label = img_path.split("/")[-2]
  scores = classifier(img_path, 
                    candidate_labels = labels_for_classification)

  top_1 = [k['label'] for k in scores[0:1]]
  top_3 = [k['label'] for k in scores[0:3]]
  top_5 = [k['label'] for k in scores[0:5]]
  top_10 = [k['label'] for k in scores[0:10]]

  if correct_label in top_1:
      top_1_count += 1

  if correct_label in top_3:
      top_3_count += 1

  if correct_label in top_5:
      top_5_count += 1
  
  if correct_label in top_10:
      top_10_count += 1

print(N)
print(top_1_count)
print(top_3_count)
print(top_5_count)
print(top_10_count)

top_1_score = (top_1_count / N ) * 100
top_3_score = (top_3_count / N ) * 100
top_5_score = (top_5_count / N ) * 100
top_10_score = (top_10_count / N ) * 100

print("top 1 score: %d / %d = %f %%" % (top_1_count, N, top_1_score))
print("top 3 score: %d / %d = %f %%" % (top_3_count, N, top_3_score))
print("top 5 score: %d / %d = %f %%" % (top_5_count, N, top_5_score))
print("top 10 score: %d / %d = %f %%" % (top_10_count, N, top_10_score))
