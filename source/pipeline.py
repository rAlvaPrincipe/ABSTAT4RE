from torch.utils import data
from property_space import PropertySpace
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
    # dim_reduc: il path è riferito a un dump, proplist è la lista di property da tenere da tenere, in fine salva in output
    def process_property_space(self, mode, dump=None, profile=None, output=None, proplist=None):
        if mode == "load":
            return PropertySpace(dump)
        elif mode == "create":
            space = PropertySpace(profile)
            space.save(output)#(self.spaces_dir + "/PS_clean_>10-freq.csv")
            return space
        elif mode == "dim_reduc":
            space = PropertySpace(dump)#(self.spaces_dir + "/PS_clean_>10-freq.csv")
            props = list()
            with open('dog_breeds.txt', 'r') as file:
                while line := file.readline().rstrip():
                    props.append(line)

            #return space.get_subspace(props, False, True, self.spaces_dir+"/PS_kbp37.csv" )
            return space.get_subspace(props, True, True, output)


    def prepare4training(self, batch, dataset, space=None, mapping=None, output=None, dump=None):
        if dump is None:
            with open(mapping) as json_file:
                props_mapping = json.load(json_file)
            labels = dataen.labels2proptensor(dataset, space, props_mapping)
            torch.save(labels, output)
        else:
            labels = torch.load(dump)

        encoded = dataen.tokenize(dataset.sentence.values)
        #dataen.check(dataset, encoded['input_ids'], labels, space, mapping)
        dataset_re = REDataset(encoded['input_ids'], encoded['attention_mask'], labels)  
        dataset_re_loader = DataLoader(dataset_re, batch_size = batch , shuffle = True)

        return dataset_re_loader


def main():
    p = Pipeline("/home/ralvaprincipe/abstat4re/")

    # ----------------------------------------- Profile ----------------------------------------------
    #profile = Profile(p.profiles_dir+"/dbpedia-2016-10", "frequency", True)
    #profile = Profile(p.profiles_dir+"/dbpedia-2016-10", "frequency", False)


    # --------------------------------------- Property Space -------------------------------------------
    #space = p.process_property_space(mode="create", profile=profile, output=p.spaces_dir+"/PS_clean_>10-freq.csv" )
    #space = p.process_property_space(mode="dim_reduc", dump=p.spaces_dir+"/PS_clean_>10-freq.csv", proplist=p.metadata_dir+"/kbp37_propertylist.txt", output=p.spaces_dir+"/PS_kbp37_nonZeroDims.csv")
    space = p.process_property_space(mode="load", dump=p.spaces_dir+"/PS_kbp37_nonZeroDims.csv")


    #-------------------------------------------- Dataset -------------------------------------------------
    #train = p.process_property_space(mode="create", path=p.datasets_dir+"/KBP37/train.txt", output=p.datasetsProc_dir+"/kbp37_CNLP_train.csv")
    #validation = p.process_property_space(mode="create", path=p.datasets_dir+"/KBP37/dev.txt", output=p.datasetsProc_dir+"/kbp37_CNLP_validation.csv")
    #test = p.process_property_space(mode="create", path=p.datasets_dir+"/KBP37/test.txt", output=p.datasetsProc_dir+"/kbp37_CNLP_test.csv")

    train = p.process_dataset(mode="load", dump=p.datasetsProc_dir+"/kbp37_CNLP_train.csv")
    validation = p.process_dataset(mode="load", dump=p.datasetsProc_dir+"/kbp37_CNLP_validation.csv")
    #  test = p.process_dataset(mode="load", dump=p.datasetsProc_dir+"/kbp37_CNLP_test.csv")
    

    #------------------------------------------- DataLoader --------------------------------------
    batch = 32
    #train_loader = p.prepare4training(batch=batch, dataset=train.df(), space=space.df(), mapping=p.metadata_dir+"kbp37_mapping.json", output=p.vec_labels+"labels-vec_kbp37_train_onlyNonZeroDims.pt")
    #val_loader = p.prepare4training(batch=batch, dataset=validation.df(), space=space.df(), mapping=p.metadata_dir+"kbp37_mapping.json", output=p.vec_labels+"labels-vec_kbp37_val_onlyNonZeroDims.pt")
    
    train_loader = p.prepare4training(batch=batch, dataset=train.df(), dump=p.vec_labels+"labels-vec_kbp37_train_onlyNonZeroDims.pt")
    val_loader = p.prepare4training(batch=batch, dataset=validation.df(), dump=p.vec_labels+"labels-vec_kbp37_val_onlyNonZeroDims.pt")
    #val_loader = p.prepare4training(batch=batch, dataset=validation.df(), dump=p.vec_labels+"labels-vec_kbp37_val_onlyNonZeroDims.pt", space=space.df(), mapping=p.metadata_dir+"kbp37_mapping.json")


    # ---------------------------------------- Classifier -----------------------------------------
    classifier = BertProjector(space.df().shape[1], True).to(p.device)
    lr = 3e-5
    epochs = 30
    criterion = CosineEmbeddingLoss() 
    optimizer = torch.optim.Adam(params = classifier.parameters(), lr=lr)

    classifier.train_loop(train_loader, val_loader, criterion, optimizer, epochs, p.device, space.df())




if __name__ == "__main__":
    main()
