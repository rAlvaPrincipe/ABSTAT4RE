import pandas as pd
import nltk
from nltk.tag.stanford import StanfordNERTagger
from nltk.tokenize import word_tokenize
from nltk.parse import CoreNLPParser
import re

class Dataset:

    def __init__(self, *args):
        if ".csv" in args[0]:
            self.read(args[0])
        else:
            self.tagger = CoreNLPParser(url='http://localhost:9666', tagtype='ner')
            self.dataset = self.parse(args[0])
            self.preprocessing()
            self.ner()


    # parse the .txt and generates a datafram for KBP37
    def parse(self, path):
        data_dict = dict()
        with open(path) as file:
            count = 0
            sentence = ""
            label = ""
            for line in file:
                count += 1
                if count == 1:
                    sentence = line[line.find("\t")+2:-2]
                elif count == 2:
                    label = line[:-1]
                elif count == 4:
                    data_dict[sentence] = label
                    count = 0
        return pd.DataFrame(list(data_dict.items()), columns=["sentence","relation"])


    def preprocessing(self):
        # Removing directionality from relations
        self.dataset["rel"] = self.dataset["relation"].str.partition("(")[0]

        # add e1 and e2 columns
        self.dataset["e1"] = self.dataset.apply (lambda row: self.get_entity(row["sentence"], True), axis = 1)
        self.dataset["e2"] = self.dataset.apply (lambda row: self.get_entity(row["sentence"], False), axis = 1)


    # adds e1_type and e2_type columns
    def ner(self):
        ner_type = "CNLP"
        count = 0
        for index in  self.dataset.index:
            sentence = self.dataset.loc[[index], 'sentence'].values[0]
            e1_Type, e2_Type = self.get_entity_types(sentence, ner_type)
            self.dataset.at[index,"e1_type_"+ner_type] = ",".join(e1_Type)
            self.dataset.at[index,"e2_type_"+ner_type] = ",".join(e2_Type)

            count +=1
            if count %500==0:
                print(count)
        

    # takes as input the sentence and a boolean used to choose the entity to return
    def get_entity(self,sentence, first):
        if first: 
            return re.search('<e1>(.*)</e1>', sentence).group(1)
        return re.search('<e2>(.*)</e2>', sentence).group(1)


    # given a sentence it resturns one set of types for e1 and one for e2
    def get_entity_types(self, sentence, ner_type):
        sentence_clean = sentence.replace("<e1>", "").replace("</e1>", "").replace("<e2>", "").replace("</e2>", "")
        try:
            # Stanford CoreNLP simple NER tagger
            if ner_type == "CNLP":
                sentence_ner = self.tagger.tag(sentence_clean.split())
                index = 0;
                e1_start, e1_end = 0, 0
                e2_start, e2_end = 0, 0
                for tok in sentence.split():
                    if tok =='<e1>':
                        e1_start = index
                    elif tok =='</e1>':
                        e1_end = index-2    
                    elif tok =='<e2>':
                        e2_start = index-2
                    elif tok =='</e2>':
                        e2_end = index-4
                    index += 1

                e1_types, e2_types= set(), set()
                for i in (e1_start, e1_end):
                    e1_types.add(sentence_ner[i][1])
                for i in (e2_start, e2_end):
                    e2_types.add(sentence_ner[i][1])

                #remove 'O' from types
                e1_types.discard('O')
                e2_types.discard('O')
                
                #TO-DO
                #anzichè usare un set usare una lista è ritornare il tag più numeroso
        except:
            print(sentence)
        return e1_types, e2_types


    def save(self, path):
        self.dataset.to_csv(path)


    def read(self, path):
        self.dataset = pd.read_csv(path, index_col=0)

    def df(self):
        return self.dataset