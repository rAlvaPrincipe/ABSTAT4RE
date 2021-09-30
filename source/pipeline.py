from torch.utils import data
from profile import Profile
from dataset import Dataset
from re_dataset import REDataset
from classifier import BertProjector
import torch
from torch import cuda
from torch.utils.data.dataloader import DataLoader
from torch.nn.modules.loss import CosineEmbeddingLoss
import dataset_encoder as dataen
import json
from torch.nn import CosineSimilarity
from property_space import PropertySpace
import pandas as pd
from check import Check as check

class Pipeline:

    def __init__(self, base):
        self.profiles_dir = base + "/data/profiles/"
        self.datasets_dir = base + "/data/datasets/"
        self.spaces_dir = base + "/data/outputs/prop_spaces/"
        self.datasetsProc_dir = base + "/data/outputs/datasets/"
        self.vec_labels = base + "/data/outputs/vec_labels/"
        self.metadata_dir = base + "/metadata/"
        self.device = 'cuda' if cuda.is_available() else 'cpu'


    def process_dataset(self, mode, path=None, output=None, dump=None):
        if mode=="load":
            return  Dataset(dump)
        elif mode=="create":
            dataset = Dataset(path)
            dataset.save()
            return dataset


    # load: carica a partire da un dump
    # create: il path è riferito alla dir del profilo e il risultato va salvato in output
    def process_property_space(self, mode, dump=None, profile=None, output=None):
        if mode == "load":
            return PropertySpace(dump)
        elif mode == "create":
            space = PropertySpace(profile)
            space.save(output)#(self.spaces_dir + "/PS_clean_>10-freq.csv")
            return space


    def prepare4training(self, batch, dataset, space=None, mappings=None, output=None, dump=None):
        rel_labels = dataset["rel"]
        if dump is None:
            vec_labels = dataen.labels2proptensor(dataset, space, mappings)
            torch.save(vec_labels, output)
        else:
            vec_labels = torch.load(dump)

        prop_labels = list()
        for i, row in dataset.iterrows():
            prop_labels.append(mappings[row["rel"]])

        encoded = dataen.tokenize(dataset.sentence.values)
        #dataen.check(dataset, encoded['input_ids'], vec_labels, space, mappings)
        #dataen.corruption_check(vec_labels, space)
        dataset_re = REDataset(encoded['input_ids'], encoded['attention_mask'], vec_labels, rel_labels, prop_labels)  
        dataset_re_loader = DataLoader(dataset_re, batch_size = batch , shuffle = True)

        return dataset_re_loader



def main():
    p = Pipeline("/home/ralvaprincipe/ABSTAT4RE/")

    ################################################### Profile ####################################################################################################
    #profile = Profile(p.profiles_dir+"/dbpedia-2016-10-full", "frequency", clean=True, artificial_props=True)

    #profile = Profile(p.profiles_dir+"/dbpedia-2016-10-full", "frequency", clean=False, artificial_props=False)
    #check.check_profile(profile.df())
    
    
    #################################################### Property Space ##############################################################################################
    #space = p.process_property_space(mode="create", profile=profile, output=p.spaces_dir+"/PS_dbp2016-full_clean_nonZeroDims.csv" )
    space = p.process_property_space(mode="load", dump=p.spaces_dir+"/PS_dbp2016-full_clean_nonZeroDims.csv")


   # check.check_PS(space.df())
    ################################################### Dataset #######################################################################################################
    train = p.process_dataset(mode="load", dump=p.datasetsProc_dir+"/kbp37_CNLP_train.csv")
    validation = p.process_dataset(mode="load", dump=p.datasetsProc_dir+"/kbp37_CNLP_validation.csv")
    test = p.process_dataset(mode="load", dump=p.datasetsProc_dir+"/kbp37_CNLP_test.csv")
    


    ################################################### DataLoader ######################################################################################################
    with open(p.metadata_dir+"kbp37_mapping.json") as json_file:
        mappings = json.load(json_file)

    
    #check.check_vec_labels_correcteness(validation.df(), space.df(), mappings)

    batch = 32
   # train_loader = p.prepare4training(batch=batch, dataset=train.df(), space=space.df(), mappings=mappings, output=p.vec_labels+"labels-vec_kbp37_train_dbp2016-full-onlyNonZeroDims.pt")
   # val_loader = p.prepare4training(batch=batch, dataset=validation.df(), space=space.df(), mappings=mappings, output=p.vec_labels+"labels-vec_kbp37_val_dbp2016-full-onlyNonZeroDims.pt")
   # test_loader = p.prepare4training(batch=batch, dataset=test.df(), space=space.df(), mappings=mappings, output=p.vec_labels+"labels-vec_kbp37_test_dbp2016-full-onlyNonZeroDims.pt")
    train_loader = p.prepare4training(batch=batch, dataset=train.df(), dump=p.vec_labels+"labels-vec_kbp37_train_dbp2016-full-onlyNonZeroDims.pt", space=space.df(), mappings=mappings)
    val_loader = p.prepare4training(batch=batch, dataset=validation.df(), dump=p.vec_labels+"labels-vec_kbp37_val_dbp2016-full-onlyNonZeroDims.pt", space=space.df(),  mappings=mappings)
   # test_loader = p.prepare4training(batch=batch, dataset=test.df(), dump=p.vec_labels+"labels-vec_kbp37_test_dbp2016-full-onlyNonZeroDims.pt", space=space.df(),  mappings=mappings)


   # check.check_label_vectorization(val_loader, space=space.df())

    ################################################### Classifier ######################################################################################################
    classifier = BertProjector(space.df().shape[1], True).to(p.device)
    lr = 3e-5
    epochs = 30
    criterion = CosineEmbeddingLoss() 
    optimizer = torch.optim.Adam(params = classifier.parameters(), lr=lr)

    classifier.train_loop(train_loader, val_loader, criterion, optimizer, epochs, p.device, space.df(), mappings)


    ########################################################################################################################################################################



if __name__ == "__main__":
    main()
