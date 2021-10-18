from torch.utils import data
from profile import Profile
from dataset import Dataset
from re_dataset import REDataset
from classifier import BertProjector, ComposedLoss
import torch
from torch import cuda
from torch.utils.data.dataloader import DataLoader
from torch.nn.modules.loss import CosineEmbeddingLoss,CrossEntropyLoss, NLLLoss
import dataset_encoder as dataen
import json
from torch.nn import CosineSimilarity
from property_space import PropertySpace
import pandas as pd
from check import Check as check
from baseline_classifier import BaselineClassifier

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
        dataset_re = REDataset(encoded['input_ids'], encoded['attention_mask'], vec_labels, rel_labels, prop_labels, mappings)  
        dataset_re_loader = DataLoader(dataset_re, batch_size = batch , shuffle = True)

        return dataset_re_loader



def main():
    p = Pipeline("/home/ralvaprincipe/ABSTAT4RE/")

    ################################################### Profile ####################################################################################################
  #  profile = Profile(p.profiles_dir+"/dbpedia-2016-10-full", "frequency", clean=True, artificial_props=True)

    #profile = Profile(p.profiles_dir+"/dbpedia-2016-10-full", "frequency", clean=False, artificial_props=False)
    #check.check_profile(profile.df())
    
    
    #################################################### Property Space ##############################################################################################
   # space = p.process_property_space(mode="create", profile=profile, output=p.spaces_dir+"/PS_dbp2016-full_clean_nonZeroDims_NoRelNew.csv" )
    #space = p.process_property_space(mode="load", dump=p.spaces_dir+"/PS_dbp2016-full_clean_nonZeroDims.csv")
    #space = p.process_property_space(mode="load", dump=p.spaces_dir+"/PS_dbp2016-full_clean_nonZeroDims_NoRelFixed.csv")
    space = p.process_property_space(mode="load", dump=p.spaces_dir+"/PS_dbp2016-full_clean_nonZeroDims_NoRelNew.csv")

  #  check.convert_pro(space.df())
  #  check.check_PS(space.df())
    ################################################### Dataset #######################################################################################################
    train = p.process_dataset(mode="load", dump=p.datasetsProc_dir+"/kbp37_CNLP_train.csv")
    validation = p.process_dataset(mode="load", dump=p.datasetsProc_dir+"/kbp37_CNLP_validation.csv")
    test = p.process_dataset(mode="load", dump=p.datasetsProc_dir+"/kbp37_CNLP_test.csv")
    


    ################################################### DataLoader ######################################################################################################
    with open(p.metadata_dir+"kbp37_mapping.json") as json_file:
        mappings = json.load(json_file)

    
    #check.check_vec_labels_correcteness(validation.df(), space.df(), mappings)

    batch = 128
    
    #train_loader = p.prepare4training(batch=batch, dataset=train.df(), space=space.df(), mappings=mappings, output=p.vec_labels+"labels-vec_kbp37_train_dbp2016-full-onlyNonZeroDims_NoRelFixed.pt")
    #val_loader = p.prepare4training(batch=batch, dataset=validation.df(), space=space.df(), mappings=mappings, output=p.vec_labels+"labels-vec_kbp37_val_dbp2016-full-onlyNonZeroDims_NoRelFixed.pt")
    #test_loader = p.prepare4training(batch=batch, dataset=test.df(), space=space.df(), mappings=mappings, output=p.vec_labels+"labels-vec_kbp37_test_dbp2016-full-onlyNonZeroDims_NoRelFixed.pt")
   # train_loader = p.prepare4training(batch=batch, dataset=train.df(), space=space.df(), mappings=mappings, output=p.vec_labels+"labels-vec_kbp37_train_dbp2016-full-onlyNonZeroDims_NoRelNew.pt")
   # val_loader = p.prepare4training(batch=batch, dataset=validation.df(), space=space.df(), mappings=mappings, output=p.vec_labels+"labels-vec_kbp37_val_dbp2016-full-onlyNonZeroDims_NoRelNew.pt")
   # test_loader = p.prepare4training(batch=batch, dataset=test.df(), space=space.df(), mappings=mappings, output=p.vec_labels+"labels-vec_kbp37_test_dbp2016-full-onlyNonZeroDims_NoRelNew.pt")

    #train_loader = p.prepare4training(batch=batch, dataset=train.df(), dump=p.vec_labels+"labels-vec_kbp37_train_dbp2016-full-onlyNonZeroDims.pt", space=space.df(), mappings=mappings)
    #val_loader = p.prepare4training(batch=batch, dataset=validation.df(), dump=p.vec_labels+"labels-vec_kbp37_val_dbp2016-full-onlyNonZeroDims.pt", space=space.df(),  mappings=mappings)
    #test_loader = p.prepare4training(batch=batch, dataset=test.df(), dump=p.vec_labels+"labels-vec_kbp37_test_dbp2016-full-onlyNonZeroDims.pt", space=space.df(),  mappings=mappings)
    #train_loader = p.prepare4training(batch=batch, dataset=train.df(), dump=p.vec_labels+"labels-vec_kbp37_train_dbp2016-full-onlyNonZeroDims_NoRelFixed.pt", space=space.df(), mappings=mappings)
    #val_loader = p.prepare4training(batch=batch, dataset=validation.df(), dump=p.vec_labels+"labels-vec_kbp37_val_dbp2016-full-onlyNonZeroDims_NoRelFixed.pt", space=space.df(),  mappings=mappings)
    #test_loader = p.prepare4training(batch=batch, dataset=test.df(), dump=p.vec_labels+"labels-vec_kbp37_test_dbp2016-full-onlyNonZeroDims_NoRelFixed.pt", space=space.df(),  mappings=mappings)
    train_loader = p.prepare4training(batch=batch, dataset=train.df(), dump=p.vec_labels+"labels-vec_kbp37_train_dbp2016-full-onlyNonZeroDims_NoRelNew.pt", space=space.df(), mappings=mappings)
    val_loader = p.prepare4training(batch=batch, dataset=validation.df(), dump=p.vec_labels+"labels-vec_kbp37_val_dbp2016-full-onlyNonZeroDims_NoRelNew.pt", space=space.df(),  mappings=mappings)
    test_loader = p.prepare4training(batch=batch, dataset=test.df(), dump=p.vec_labels+"labels-vec_kbp37_test_dbp2016-full-onlyNonZeroDims_NoRelNew.pt", space=space.df(),  mappings=mappings)

   # check.check_label_vectorization(val_loader, space=space.df())


    ################################################### Baselilne ######################################################################################################
    baseline = BaselineClassifier(space.df().shape[1], True).to(p.device)
    lr = 3e-5
    epochs = 70
    criterion = CrossEntropyLoss()
    #criterion = NLLLoss()
    optimizer = torch.optim.Adam(params = baseline.parameters(), lr=lr)
    patience = 1

    baseline.train_loop(train_loader, val_loader, criterion, optimizer, epochs, patience, p.device, space.df(), mappings, "baseline")


    ################################################### Classifier ######################################################################################################

    abstat4re = BertProjector(space.df().shape[1], True).to(p.device)
    lr = 3e-5
    epochs = 70
    criterion = ComposedLoss(p.device) 
    optimizer = torch.optim.Adam(params = abstat4re.parameters(), lr=lr)
    patience = 1
    
    abstat4re.train_loop(train_loader, val_loader, criterion, optimizer, epochs, patience, p.device, space.df(), mappings, "abstat4re")
    

    ###################################################### TEST ########################################################################################################
    
    baseline = BaselineClassifier(space.df().shape[1], True).to(p.device)
    baseline.load_state_dict(torch.load('results/baseline.pt'))
    baseline.test_loop(test_loader, p.device, space.df(), mappings)

    abstat4re = BertProjector(space.df().shape[1], True).to(p.device)
    abstat4re.load_state_dict(torch.load('/resultsabstat4re.pt'))
    abstat4re.test_loop(test_loader, p.device, space.df(), mappings)



if __name__ == "__main__":
    main()
