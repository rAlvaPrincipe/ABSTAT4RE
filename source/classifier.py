from pandas.core.frame import DataFrame
from transformers import DistilBertModel
from torch.nn import Module, Linear, ReLU, Sigmoid
import torch
from torch.nn import CosineSimilarity

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
        self.fc3 = Layer(768, n_classes, ReLU) 
   
   
    def forward(self, tokenized_sents, attn_masks):
        x = self.bert(tokenized_sents, attn_masks)
        x = x['last_hidden_state'][:, 0]
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    

    def train_loop(self, train_loader, val_loader, criterion, optimizer, epochs, device, space: DataFrame):
        train_losses, val_losses = [], []
        for e in range(epochs):
            train_epoch_loss = 0
            for sentences, masks, labels in train_loader:
            #    iz=0
               # print(sentences.shape)
               # print(masks.shape)
               # print(labels.shape)
                current_batch_size = sentences.shape[0]
                outputs = self(sentences.to(device), masks.to(device))
                optimizer.zero_grad()
                loss = criterion(outputs, labels.to(device), torch.ones(current_batch_size).to(device))
                loss.backward()
                optimizer.step()
                train_epoch_loss += loss  
            else:
                val_epoch_loss = 0
                val_epoch_acc = 0
                tot_epoch_corects = 0
                cos = CosineSimilarity(dim=0)
                with torch.no_grad():
                    self.eval()
                    for sentences, masks, labels in val_loader:
                        current_batch_size = sentences.shape[0]
                        sentences = sentences.to(device)
                        masks = masks.to(device)
                        labels = labels.to(device)
                        ones = torch.ones(current_batch_size).to(device)

                        outputs = self(sentences, masks)
                        loss = criterion(outputs, labels, ones)
                        val_epoch_loss += loss

                        # calculate accuracy
                        batch_corrects = 0
                        for i in range(current_batch_size):
                            max_sim=0
                            closest_candidate=None
                            for property, row_space in space.iterrows():
                                vector = torch.tensor(space.loc[property]).to(device) 
                                sim = cos(outputs[i], vector).item()
                                if sim >= max_sim:
                                    closest_candidate = vector
                                    max_sim = sim

                            if torch.equal(labels[i], closest_candidate):
                                batch_corrects += 1
                    
                        tot_epoch_corects += batch_corrects
                        accuracy = batch_corrects/current_batch_size
                        val_epoch_acc += accuracy

                print(tot_epoch_corects)
                self.train()
                train_loss = train_epoch_loss/len(train_loader)
                val_loss = val_epoch_loss/len(val_loader) 
                val_acc = val_epoch_acc/len(val_loader) 
                train_losses.append(train_loss)
                val_losses.append(val_loss) 

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                    "Training Loss: {:.3f}.. ".format(train_loss),
                    "Validation Loss: {:.3f}.. ".format(val_loss),
                    "Validation Acc: {:.3f}.. ".format(val_acc))
                    

    



        #esperimento: anzichè confrontarlo solo con i vettori die 20 predicati che ti interssano, confronta con tutti, e trova a chi si avvicina di più