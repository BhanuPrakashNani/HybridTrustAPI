import nltk
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Autonomous cars shift insurance liability toward manufacturers")



!pip install --pre spacy-nightly

!python -m spacy download en_core_web_trf
!python -m spacy download en_core_web_sm

import spacy
from spacy import displacy
from nltk import Tree


nlp = spacy.load('en_core_web_trf')

doc = nlp("I need a Cloud Service Provider with decent performance. I also want the uptime is high as possible and the downtime to be very low. The application update frequency can be pretty moderate and the fault tolerance capability very low.")

def generate_svg(doc):
  from pathlib import Path

  output_path = Path("test.svg")
  svg = displacy.render(doc, style='dep')
  with output_path.open("w", encoding="utf-8") as fh:
      fh.write(svg)

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_


[to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]
generate_svg(doc)



import spacy
from spacy import displacy
from nltk import Tree
import numpy as np

def parse_request(text):
  qos_params = ["uptime", "downtime", "application update frequency", "fault tolerance capability"]
  compounded_words = []
  level_indicators = []
  nlp = spacy.load('en_core_web_trf')
  doc = nlp(text)
  mapping = {}
  parse = [(t.i, t, t.pos_, t.tag_, t.dep_, t.head) for t in doc]
  # print(parse)
  for t in doc:
      if(t.pos_ == "ADJ"):
        tmp = ""
        for x in t.children:
          if(x.dep_=="advmod"):
            tmp+=" "+str(x)
        tmp+=" "+str(t)
        level_indicators.append([tmp.strip(),t.i])

  for t in doc:
    if(t.pos_ == "NOUN"):
      tmp = ""
      # print("main :"+str(t))
      if(str(t)=="capability"):
        for x in t.children:
          if(x.dep_=="compound"):
            for y in x.children:
              if(y.dep_ == "compound"):
                compounded_words.append([f"{str(y)} {str(x)} {str(t)}", t.i])
      else:
        for x in t.children:
          if(x.dep_ =="compound"):
            for y in x.children:
              if(y.dep_ =="compound"):
                tmp+=" "+str(y)
            tmp+=" "+str(x)
        tmp+=" "+str(t)
        compounded_words.append([tmp.strip(), t.i])

  for word in compounded_words:
    if(word[0] in qos_params):
      t = doc[word[1]]
      # print(str(word[0]), (t.i, t, t.pos_, t.tag_, t.dep_, t.head))
      for x in doc[word[1]].children:
        if(x.pos_ == "ADJ"):
          key = ""
          for w in level_indicators:
            if(x.i == w[1]):
              key = w[0]
          mapping[str(word[0])] = key
        # print((x.i, x, x.pos_, x.tag_, x.dep_, x.head))

  # for cases where ADJs are not in deps of NOUN but as a chain with AUX to NOUNS
  if(len(mapping)!=4):
    for adj in level_indicators:
      d = doc[adj[1]]
      print("main:"+str(d)+" with level: "+adj[0])
      parent = d.head
      for sibling in d.children:
        if(sibling.pos_ == "NOUN"):
          print(sibling, sibling.i)
          key =""
          for word in compounded_words:
            if(word[1] == sibling.i):
              key  = word[0]
          if(key in qos_params and key not in mapping.keys()):
            mapping[key] = adj[0] 
      for child in parent.children:
        if(child.pos_ == "NOUN"):
          print(child, child.i)
          key =""
          for word in compounded_words:
            if(word[1] == child.i):
              key  = word[0]
          if(key in qos_params and key not in mapping.keys()):
            mapping[key] = adj[0] 
  print(mapping)
  print(compounded_words)
  print(level_indicators)

parse_request("I need a csp with good uptime with very less downtime and medium application update frequency")

text = "I need a csp with good uptime with very less downtime, medium application update frequency and high fault tolerance capability."
parse_request(text)

text = "I need a Cloud Service Provider with decent performance. I also want the uptime to be high as possible and the downtime to be very low. The application update frequency can be pretty moderate and the fault tolerance capability can be low."
parse_request(text)

text = "I need a Cloud Service Provider with decent performance. I also want the uptime is high as possible and the downtime to be very low. The application update frequency can be pretty moderate and the fault tolerance capability very low."
# doc = nlp(text)
# parse = [(t.i, t, t.pos_, t.tag_, t.dep_, t.head) for t in doc]
# parse
parse_request(text)

sentences = ["""I need a csp with good uptime with very less downtime and medium application update frequency""",
             """ I need a Cloud Service Provider with decent performance. I also want the uptime to be high as possible and the downtime to be very low. The application update frequency can be moderate and the fault tolerance capability can be low.""",
             """CSP with high uptime low downtime least application update frequency and highest fault tolerance capability"""
             ]
for sentence in sentences:
  tokens = nltk.word_tokenize(sentence)
  doc = nlp(sentence)
  print(tokens)
  # generate_svg(doc)
  tagged = nltk.pos_tag(tokens)

  print(tagged)
  dg.tree().pprint()

