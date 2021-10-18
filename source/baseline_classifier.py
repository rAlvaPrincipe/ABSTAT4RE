from torch.nn.modules.activation import LogSoftmax, Softmax, LogSoftmax
from transformers import DistilBertModel
from torch.nn import Module, Linear, ReLU, Sigmoid, Tanh, Dropout
import torch
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn import metrics
from transformers import DistilBertModelWithHeads
from transformers import AdapterConfig, DistilBertConfig
from pytorchtools import EarlyStopping
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt 

class Layer(Module):
    def __init__(self, inp, out, act):
        super().__init__()
        self.fc = Linear(inp, out)
        self.act = act()
        
    def forward(self, x):
        return self.act(self.fc(x))


class BaselineClassifier(Module):
    def __init__(self, n_classes, freeze_bert):
        super().__init__()

    # DISTILBERT + Adapters
        self.bert = DistilBertModelWithHeads.from_pretrained("distilbert-base-uncased")
        config = AdapterConfig.load("pfeiffer", reduction_factor=16)
        self.bert.add_adapter("prova",config=config)
        self.bert.train_adapter(["prova"])

      # DISTILBERT standard
     #   self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
     #   if freeze_bert:
     #       for param in self.bert.parameters():
     #           param.requires_grad = False

        self.fc1 = Layer(768, 256, ReLU)
        self.fc2 = Layer(256, 64, ReLU)
        self.fc3 = Linear(64, 19) 
       # self.logsoftmax = LogSoftmax(dim=1)
   
      
    def forward(self, tokenized_sents, attn_masks):
        x = self.bert(tokenized_sents, attn_masks)
        x = x['last_hidden_state'][:, 0]
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
      #  x = self.logsoftmax(x)
        return x
    

    def train_loop(self, train_loader, val_loader, criterion, optimizer, epochs, patience, device, space, mappings, name):
        early_stopping = EarlyStopping(patience=patience, verbose=True, path="results/" + name+".pt")

        for e in tqdm(range(epochs)):
            train_epoch_loss = 0

            self.train()
            for sentences, masks, vec_labels, rel_labels, prop_labels, int_labels, no_relation_tensor, attribute_tensor in train_loader:
                current_batch_size = sentences.shape[0]
                log_probs = self(sentences.to(device), masks.to(device))
                optimizer.zero_grad()
                loss = criterion(log_probs, int_labels.to(device))
                loss.backward()
                optimizer.step()
                train_epoch_loss += loss  

            self.eval()
            if (e + 1) % 1 == 0:
                val_epoch_loss = 0
                epoch_labels, epoch_predictions = tuple(), tuple()
                with torch.no_grad():
                    for sentences, masks, vec_labels, rel_labels, prop_labels, int_labels, no_relation_tensor, attribute_tensor in val_loader:
                        current_batch_size = sentences.shape[0]
                        log_probs = self(sentences.to(device), masks.to(device))
                        loss = criterion(log_probs, int_labels.to(device))

                        # getting prediction labels
                        probs = torch.exp(log_probs)
                        top_prob, top_class = probs.topk(1, dim=1)
                        top_class = top_class.reshape(current_batch_size)
                        predictions = self.integer_to_labels(top_class, mappings)

                        # saving batch predictions/stats for later
                        epoch_labels += rel_labels
                        epoch_predictions += tuple(predictions)

                        val_epoch_loss += loss  
            
                accuracy, macro_f1 = self.print_metrics(epoch_labels, epoch_predictions, mappings)
                self.show_confusion_matrix(epoch_labels, epoch_predictions, mappings)
      
                train_loss = train_epoch_loss/len(train_loader)
                val_loss = val_epoch_loss/len(val_loader) 

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                    "Training Loss: {:.3f}.. ".format(train_loss),
                    "Validation Loss: {:.3f}.. ".format(val_loss))
              
              
                # early_stopping needs the validation loss to check if it has decresed, 
                # and if it has, it will make a checkpoint of the current model
                early_stopping(-macro_f1, self) # here i use f1. I put the - before to keep unchanged the function code
                
                if early_stopping.early_stop:
                    print("Early stopping")
                    break



    # convert integer_labels 
    def integer_to_labels(self, integer_labels, mappings):
        mappings = dict(sorted(mappings.items(), key=lambda item: item[1]))
        vals = list(mappings.values())
        reverse_mapping = inv_map = {v: k for k, v in mappings.items()}

        labels = list()
        for index in integer_labels:
            #labels.append(vals[index])
            labels.append(reverse_mapping[vals[index]])
        return labels


    def print_metrics(self, labels:list, predictions, mappings):
        print(metrics.classification_report(labels, predictions, labels=list(mappings)))
        report = metrics.classification_report(labels, predictions, labels=list(mappings), output_dict=True)
        accuracy = report['accuracy']
        macro_f1 = report['macro avg']['f1-score']
        print("accuracy: {}".format(accuracy))
        print("F1: {}".format(macro_f1))
        return accuracy, macro_f1



    def show_confusion_matrix(self, labels:list, predictions, mappings):
        cm = confusion_matrix(y_true=labels, y_pred=predictions, labels=list(mappings))
        #print(cm)
        cmd_obj = ConfusionMatrixDisplay(cm, display_labels=list(mappings))
        cmd_obj.plot()
        cmd_obj.ax_.set(
                        title='Confusion Matrix', 
                        xlabel='Predicted Properties', 
                        ylabel='Actual Properties')
        plt.show()
        plt.xticks(rotation=70, ha="right")
        plt.savefig('results/baseline_cm.png', bbox_inches='tight')



    def test_loop(self, test_laoder, device, space: DataFrame, mappings):
        self.eval()
        epoch_labels, epoch_predictions = tuple(), tuple()
        with torch.no_grad():
            for sentences, masks, vec_labels, rel_labels, prop_labels, int_labels, no_relation_tensor, attribute_tensor in test_laoder:
                current_batch_size = sentences.shape[0]
                log_probs = self(sentences.to(device), masks.to(device))
                # getting prediction labels
                probs = torch.exp(log_probs)
                top_prob, top_class = probs.topk(1, dim=1)
                top_class = top_class.reshape(current_batch_size)
                predictions = self.integer_to_labels(top_class, mappings)

                epoch_labels += rel_labels
                epoch_predictions += tuple(predictions)

        self.print_metrics(epoch_labels, epoch_predictions, mappings)
        self.show_confusion_matrix(epoch_labels, epoch_predictions, mappings)

