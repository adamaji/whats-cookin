usage = """
Usage: python sentence_classifier.py [ingredients text file] \
       [instructions text file] [background text file] \
       [dir with video descriptions] [output directory]

Naive bayes classifier to distinguish recipe sentences from extraneous background 
information sentences. Requires each input file to have sentences from the class 
separated by newlines.
"""

import sys
import os
import numpy as np
import random
import nltk.data
from io import open
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer

INGREDIENT_LABEL = 0
INSTRUCTION_LABEL = 1
BACKGROUND_LABEL = 2

def get_all_sentences(path):
  strings = []
  with open(path, "r", encoding="utf-8") as f:
    for line in f:
      strings.append(line)
  return strings

def train_and_classify(ingredient_text_path, instruction_text_path, bg_text_path, test_path, out_path):
  ing = get_all_sentences(ingredient_text_path)
  ins = get_all_sentences(instruction_text_path)
  bg = get_all_sentences(bg_text_path)

  random.shuffle(ing)
  random.shuffle(ins)
  random.shuffle(bg)

  div_ing = int(0.9*len(ing))
  div_ins = int(0.9*len(ins))
  div_bg = int(0.9*len(bg))

  x_train =  ing[:div_ing] + ins[:div_ins] + bg [:div_bg]
  x_test =  ing[div_ing:] + ins[div_ins:] + bg [div_bg:]
  y_train = [INGREDIENT_LABEL] * div_ing + [INSTRUCTION_LABEL] * div_ins + [BACKGROUND_LABEL] * div_bg
  y_test = np.asarray([INGREDIENT_LABEL] * (len(ing) - div_ing) + [INSTRUCTION_LABEL] * (len(ins) - div_ins) + [BACKGROUND_LABEL] * (len(bg) - div_bg))

  from sklearn.feature_extraction.text import CountVectorizer
  cv = CountVectorizer()
  x_train_cv = cv.fit_transform(x_train)
  tf_transformer_x_train = TfidfTransformer(use_idf=True)
  x_train_tf = tf_transformer_x_train.fit_transform(x_train_cv)

  classifier = MultinomialNB().fit(x_train_tf, y_train)

  x_test_cv = cv.transform(x_test)
  tf_transformer_x_test = TfidfTransformer(use_idf=True)
  x_test_tf = tf_transformer_x_test.fit_transform(x_test_cv)
  predicted = classifier.predict(x_test_tf)
  print np.mean(predicted == y_test)

  sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

  # classify on the video descriptions
  for fn in os.listdir(test_path):
    text = ""
    with open(os.path.join(test_path, fn), "r", encoding="utf-8") as f:
      text = f.read()
    sents = sent_detector.tokenize(text.strip().lower())
    sents_cv = cv.transform(sents)
    tf_transformer = TfidfTransformer(use_idf=True)
    sents_tf = tf_transformer.fit_transform(sents_cv)
    predicts = classifier.predict(sents_tf)
    with open(os.path.join(out_path, fn + ".clf"), "w", encoding="utf-8") as f:
      for i in range(len(sents)):
        if predicts[i] == INSTRUCTION_LABEL:
          instruction = sents[i].strip("\n")
          if instruction != "":
            f.write(instruction + "\n")

if __name__ == "__main__":
  if len(sys.argv) != 6:
    print usage
    sys.exit(-1)

  train_and_classify(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])