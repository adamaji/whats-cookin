# Searches video description for sentence which best corresponds
# to the action/object pair from the Whats Cookin dataset 

usage = "python align_action_object_to_sentence.py [video csv] [path to video description dir] [output file]"
GLOVE_MODEL_PATH = "./glove.42B.300d.txt"

import os
import sys
import nltk.data
import numpy as np
from io import open
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

def load_glove_model(fn):
  print("loading glove model...")
  model = {}
  with open(fn, "r", encoding="utf-8") as f:
    for line in f:
      split = line.split()
      word = split[0]
      embedding = [float(val) for val in split[1:]]
      model[word] = np.asarray(embedding)
  return model

def compute_euclidean_distance(a, b):
  return np.linalg.norm(a-b)

def align_action_object_to_sentence(video_csv, desc_dir, output_fn):
  vec_model = load_glove_model(GLOVE_MODEL_PATH)
  with open(video_csv, "r", encoding="utf-8") as csv, open(output_fn, "w", encoding="utf-8") as out_f:
    # do the alignment for each video segment
    for line in csv:
      csv_split = line.split(",")
      youtube_id = csv_split[0]
      ms_id = csv_split[1]
      act_str = csv_split[3].lower()
      obj_str = csv_split[4].lower()
      tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
      lemmatizer = WordNetLemmatizer()
      matches = []      
      with open(os.path.join(desc_dir, ".".join([youtube_id,"description"])), "r", encoding="utf-8") as f:
        text = f.read()
        split = text.split("\n")
        sentences = []
        for s in split:
          sentences += tokenizer.tokenize(s.strip().lower())

        # check sentences in description for lemmatized action
        for sent in sentences:
          sent_match = False
          words = word_tokenize(sent)
          for word in words:
            lemma = lemmatizer.lemmatize(word)
            if lemma == act_str:
              sent_match = True
          if sent_match:
            matches.append(sent)
        
        # then check for similar objects
        if len(matches) > 1 and obj_str != "":
          min_dist = None
          min_sent = None
          for sent in matches:
            words = word_tokenize(sent)
            for word in words:
              if word in vec_model:
                dist = compute_euclidean_distance(vec_model[word], vec_model[obj_str])
                if min_dist == None or dist < min_dist:
                  min_dist = dist
                  min_sent = sent
          matches = [min_sent]

      # take the first occurence after matching obj/act
      if len(matches) > 0:
        out_f.write(".".join([youtube_id,ms_id]) + "\t" + matches[0] + "\n")

if __name__ == "__main__":
  if len(sys.argv) < 4:
    print usage
    sys.exit(-1)
  align_action_object_to_sentence(sys.argv[1], sys.argv[2], sys.argv[3])