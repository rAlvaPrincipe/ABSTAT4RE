from torch.nn.modules.activation import LogSoftmax, Softmax, LogSoftmax
from transformers import DistilBertModel
from torch.nn import Module, Linear, ReLU, Sigmoid, Tanh, Dropout
import torch
import pandas as pd
from sklearn import metrics

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

        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

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
    

    def train_loop(self, train_loader, val_loader, criterion, optimizer, epochs, device, space, mappings):
        for e in range(epochs):
            print("epoch {}".format(e+1))
            train_epoch_loss = 0

            for sentences, masks, vec_labels, rel_labels, prop_labels, int_labels in train_loader:
                current_batch_size = sentences.shape[0]
                log_probs = self(sentences.to(device), masks.to(device))
                optimizer.zero_grad()
                loss = criterion(log_probs, int_labels.to(device))
                loss.backward()
                optimizer.step()
                train_epoch_loss += loss  
            
            if (e + 1) % 2 == 0:
                val_epoch_loss, val_epoch_acc, tot_epoch_corects = 0, 0, 0
                epoch_labels, epoch_predictions = tuple(), tuple()
              
                with torch.no_grad():
                    self.eval()
                    for sentences, masks, vec_labels, rel_labels, prop_labels, int_labels in val_loader:
                        current_batch_size = sentences.shape[0]
                        log_probs = self(sentences.to(device), masks.to(device))
                        loss = criterion(log_probs, int_labels.to(device))

                        # getting prediction labels
                        probs = torch.exp(log_probs)
                        top_prob, top_class = probs.topk(1, dim=1)
                        top_class = top_class.reshape(current_batch_size)
                        predictions = self.integer_to_labels(top_class, mappings)

                        # saving batch predictions/stats for later
                        epoch_labels += prop_labels
                        epoch_predictions += tuple(predictions)

                        val_epoch_loss += loss  
            
                print(tot_epoch_corects)
                self.print_metrics(epoch_labels, epoch_predictions)
                self.train()
                train_loss = train_epoch_loss/len(train_loader)
                val_loss = val_epoch_loss/len(val_loader) 
                
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                    "Training Loss: {:.3f}.. ".format(train_loss),
                    "Validation Loss: {:.3f}.. ".format(val_loss))




    # convert integer_labels 
    def integer_to_labels(self, integer_labels, mappings):
        mappings = dict(sorted(mappings.items(), key=lambda item: item[1]))
        vals = list(mappings.values())

        labels = list()
        for index in integer_labels:
            labels.append(vals[index])
        return labels


    def print_metrics(self, labels:list, predictions):
        print("ground truth labels: {}".format(set(labels)))
        print("prediction labels: {}".format(set(predictions)))
        print(metrics.classification_report(labels, predictions, labels=list(set(labels))))
        print("accuracy: {}".format(metrics.accuracy_score(labels, predictions)))

