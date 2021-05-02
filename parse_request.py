import spacy
from spacy import displacy
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
      parent = d.head
      for sibling in d.children:
        if(sibling.pos_ == "NOUN"):
          key =""
          for word in compounded_words:
            if(word[1] == sibling.i):
              key  = word[0]
          if(key in qos_params and key not in mapping.keys()):
            mapping[key] = adj[0] 
      for child in parent.children:
        if(child.pos_ == "NOUN"):
          key =""
          for word in compounded_words:
            if(word[1] == child.i):
              key  = word[0]
          if(key in qos_params and key not in mapping.keys()):
            mapping[key] = adj[0]
  return mapping

text = "I need a Cloud Service Provider with decent performance. I also want the uptime to be high as possible and the downtime to be very low. The application update frequency can be pretty moderate and the fault tolerance capability can be low."
print(parse_request(text))