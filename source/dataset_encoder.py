from pandas.core.frame import DataFrame
import torch
from transformers import DistilBertTokenizer
import pandas as pd
from torch.nn import CosineSimilarity
import json

# LABELS TENSOR GENERATOR
# based on mappings kbp37-->dbpedia a tensor with the label (property vectors) for each sentence is crated
# final size labels: (|sentences|, dim_property_space)
def labels2proptensor(dataset, space, props_mapping):
    temp = torch.Tensor()
    size_buff = 200
    for i, row in dataset.iterrows():
        index = props_mapping[row["rel"]]
        label = torch.tensor(space.loc[index])
        temp = torch.cat([temp, label], dim=0)

        if(i % size_buff  == 0 and i!=0):
            temp = temp.view(-1, space.shape[1])
            if(i == size_buff):
                labels = temp
            else:
                labels = torch.vstack((labels, temp))
            temp = torch.Tensor()
            print(i)
    temp = temp.view(-1, space.shape[1])
    labels = torch.vstack((labels, temp))  
    return labels



def tokenize(sentences):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    return  tokenizer.batch_encode_plus(
                sentences,
                add_special_tokens=True,
                return_attention_mask=True,
                pad_to_max_length=True,
                max_length=512,
                return_tensors='pt'
            )


# it checks if endoing of sentences and labels and association sentence-labels are ok
def check(dataset:DataFrame, encoded, labels:torch.Tensor, space: DataFrame, mappings):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    with open(mappings) as json_file:
        props_mapping = json.load(json_file)
    
    sentence_check = True
    for index in range(dataset.shape[0]):
        encoded_posteriori = tokenizer.encode(dataset.loc[index,"sentence"], add_special_tokens=True,return_attention_mask=True,pad_to_max_length=True,max_length=512,return_tensors='pt').view(512)
        
        if not torch.equal(encoded[index], encoded_posteriori):
            sentence_check = False
            print("error index:{}".format(index))

    if sentence_check:
        print("sentences are ok")


    label_check = True
    for index, row_dataset in dataset.iterrows():

        for property, row_space in space.iterrows():
            vector = torch.tensor(space.loc[property])        
            if torch.equal(vector, labels[index]):
                prop  = property
        if not prop == props_mapping[row_dataset["rel"]]:
            sentence_check = False
    if label_check:
        print("labels are ok")
        
    
# si assicura che ogni label_vec corrisponda a un solo predicato nello spazio delle property
def corruption_check(vec_labels, space: DataFrame):
    cos = CosineSimilarity(dim=0)
    count = 0
    for index in range(vec_labels.shape[0]):
        ok = False
        for property, row in space.iterrows():
            vector = torch.tensor(space.loc[property]) 
            if cos(vec_labels[index], vector) == 1:
                vec_labels[index]
                ok = True
        
        print(torch.sum(vec_labels[index]))
        if not ok:
            print("ERRORE")
            count +=1
        else:
            print("OK")
    print(count)