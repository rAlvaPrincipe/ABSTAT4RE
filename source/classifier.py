from numpy import tanh
from pandas.core.frame import DataFrame
from transformers import DistilBertModel
from torch.nn import Module, Linear, ReLU, Sigmoid, Tanh
import torch
from torch.nn import CosineSimilarity
from sklearn import metrics
import json

class Layer(Module):
    def __init__(self, inp, out, act):
        super().__init__()
        self.fc = Linear(inp, out)
        self.act = act()
        
    def forward(self, x):
        return self.act(self.fc(x))


class BertProjector(Module):
    def __init__(self, n_classes, freeze_bert):
        super().__init__()

        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.fc1 = Layer(768, 768, ReLU)
        self.fc2 = Layer(768, 768, ReLU)
        self.fc3 = Layer(768, n_classes, Tanh) 
   
   
    def forward(self, tokenized_sents, attn_masks):
        x = self.bert(tokenized_sents, attn_masks)
        x = x['last_hidden_state'][:, 0]
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    

    def train_loop(self, train_loader, val_loader, criterion, optimizer, epochs, device, space: DataFrame, mappings):
        for e in range(epochs):
            print("epoch {}".format(e+1))
            train_epoch_loss = 0
          #  count =0
            for sentences, masks, vec_labels, labels in train_loader:
            #    iz=0
                current_batch_size = sentences.shape[0]
                outputs = self(sentences.to(device), masks.to(device))
                optimizer.zero_grad()
                loss = criterion(outputs, vec_labels.to(device), torch.ones(current_batch_size).to(device))
                loss.backward()
                optimizer.step()
                train_epoch_loss += loss  
            #    count += 1
            #    if count == 3:
             #       break
            
            if e > 15:
                val_epoch_loss, val_epoch_acc, tot_epoch_corects = 0, 0, 0
                epoch_labels, epoch_predictions = tuple(), tuple()
                with torch.no_grad():
                    self.eval()
                    #count=0
                    for sentences, masks, vec_labels, labels in val_loader:
                        outputs = self(sentences.to(device), masks.to(device))
                        loss = criterion(outputs, vec_labels.to(device), torch.ones(vec_labels.shape[0]).to(device))

                    # batch_corrects = self.calculate_corrects(outputs, space, vec_labels.to(device), device) 
                    # print("---accuracy con batchcorrects: {}".format(batch_corrects/vec_labels.shape[0]))   
                    # labels_ricavati = self.convert(vec_labels.to(device), space, device)
                        #print(labels_ricavati)

                        #self.print_metrics(labels_ricavati, self.convert(outputs, space, device))
                        epoch_labels +=  self.convert(vec_labels.to(device), space, device)
                        epoch_predictions += self.convert(outputs, space, device)


                    # tot_epoch_corects += batch_corrects
                    #  accuracy = batch_corrects/vec_labels.shape[0]
                    #  val_epoch_acc += accuracy
                        val_epoch_loss += loss   
                    #  count +=1
                    #  if count == 3:
                    #      break

                print(tot_epoch_corects)
                self.print_metrics(epoch_labels, epoch_predictions)
                self.train()
                train_loss = train_epoch_loss/len(train_loader)
                val_loss = val_epoch_loss/len(val_loader) 
            #   val_acc = val_epoch_acc/len(val_loader) 

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                    "Training Loss: {:.3f}.. ".format(train_loss),
                    "Validation Loss: {:.3f}.. ".format(val_loss))
                # "Validation Acc: {:.3f}.. ".format(val_acc))
        


    def convert(self, outputs, space, device):
        predictions = tuple()
        cos = CosineSimilarity(dim=0)
        closest_candidate=None
        for output in outputs:
            max_sim = -1
            for property, row_space in space.iterrows():
                vector = torch.tensor(space.loc[property]).to(device)
                sim = cos(output, vector).item()
                if sim >= max_sim:
                    closest_candidate = property
                    max_sim = sim
        #    print(max_sim)
            predictions += (closest_candidate,)

        return predictions


    def calculate_corrects(self, outputs, space, vec_labels, device):
        cos = CosineSimilarity(dim=0)
        batch_corrects = 0
        for i in range(vec_labels.shape[0]):
            max_sim=0
            closest_candidate=None
            for property, row_space in space.iterrows():
                vector = torch.tensor(space.loc[property]).to(device) 
                sim = cos(outputs[i], vector).item()
                if sim >= max_sim:
                    closest_candidate = vector
                    max_sim = sim
            if torch.equal(vec_labels[i], closest_candidate):
                batch_corrects += 1
        return batch_corrects



    def print_metrics(self, labels:list, predictions):
        print(metrics.classification_report(labels, predictions, labels=list(set(labels))))
        print("accuracy: {}".format(metrics.accuracy_score(labels, predictions)))

        corrects =0
        for label, prediction in zip(labels, predictions):
            if label==prediction:
                corrects += 1
        print("rechecked accuracy: {}".format(corrects/len(labels)))  



#    def tracker(self, labels, outputs, space, tracks:DataFrame):
#        tracks = dict()
#
#        for index, output in enumerate(outputs):
#            prediction = self.closest_property(output, space)
#
#            if prediction == labels[index]:
#                tracks[tracks["class"]==labels[index]]["TP"] +=1          #TP
#            else:
#                a=0
#            #FP
#            #TN
#            #FN
#        return tracks



    # given a prediction it returns the closest class
 #   def closest_property(self, prediction, space):
 #       cos = CosineSimilarity(dim=0)
 #       max_sim=0
 #       for property, row_space in space.iterrows():
 #           vector = torch.tensor(space.loc[property])
 #           sim = cos(prediction, vector).item()
 #           if sim >= max_sim:
 #               closest_candidate = property
 #               max_sim = sim
 #       return closest_candidate

        #esperimento: anzichè confrontarlo solo con i vettori die 20 predicati che ti interssano, confronta con tutti, e trova a chi si avvicina di più