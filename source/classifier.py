from numpy import tanh
from pandas.core.frame import DataFrame
from torch.nn.modules.loss import BCELoss, CosineEmbeddingLoss
from transformers import DistilBertModel
from torch.nn import Module, Linear, ReLU, Sigmoid, Tanh
import torch
from torch.nn import CosineSimilarity
from sklearn import metrics
import json
from transformers import DistilBertModelWithHeads
from transformers import AdapterConfig, DistilBertConfig
from tqdm import tqdm

class ComposedLoss(Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.projection_loss = CosineEmbeddingLoss(reduction='none')
        self.attribute_loss = BCELoss()
        self.device = device
    
    def forward(self, projected_vectors, predicted_attributes, groundtruth_vectors, groundtruth_attributes, no_relation_tensor):
        zeros = torch.zeros(no_relation_tensor.shape, dtype=torch.double).to(self.device)

        # calculate loss for every sample despite "no_relation"
        projection_loss_value = self.projection_loss(projected_vectors, groundtruth_vectors, torch.ones(projected_vectors.shape[0]).to(self.device))
        
        # remove loss for samples that are "no_relaion"
        projection_loss_value_with_no_relation = torch.where(no_relation_tensor.to(self.device), zeros, projection_loss_value)

        # obtain a single scalar for the losses of the batch
        average_projection_loss = torch.mean(projection_loss_value_with_no_relation)

        # callcualte loss for the "no_relation" classifier
        attribute_loss_value = self.attribute_loss(predicted_attributes.squeeze(), groundtruth_attributes.to(self.device))

        return average_projection_loss + attribute_loss_value, average_projection_loss, attribute_loss_value


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

    # DISTILBERT + Adapters
        self.bert = DistilBertModelWithHeads.from_pretrained("distilbert-base-uncased")
        config = AdapterConfig.load("pfeiffer", reduction_factor=16)
        self.bert.add_adapter("prova",config=config)
        self.bert.train_adapter(["prova"])

      # DISTILBERT standard
      #  self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
      #  if freeze_bert:
      #      for param in self.bert.parameters():
      #          param.requires_grad = False

        self.fc1 = Layer(768, 768, ReLU)
        self.fc2 = Layer(768, 768, ReLU)
        self.fc3 = Layer(768, n_classes, ReLU)

        self.relation_attribute = Layer(768, 1, Sigmoid) # Layer(768, 3, Sigmoid) quando useremo la direzionalità   
   
    def forward(self, tokenized_sents, attn_masks):
        x = self.bert(tokenized_sents, attn_masks)
        cls_encoding = x['last_hidden_state'][:, 0]
        
        # compute projection
        x = self.fc1(cls_encoding)
        x = self.fc2(x)
        projection = self.fc3(x)

        # compute attribute
        attribute = self.relation_attribute(cls_encoding)

        return projection, attribute
    

    def train_loop(self, train_loader, val_loader, criterion, optimizer, epochs, device, space: DataFrame, mappings):
        for e in tqdm(range(epochs)):
            train_epoch_loss = 0
            train_projection_loss = 0
            train_attribute_loss = 0

            self.train()
            for sentences, masks, vec_labels, rel_labels, prop_labels, int_labels, no_relation_tensor, attribute_tensor in train_loader:
                current_batch_size = sentences.shape[0]
                projection, attribute = self(sentences.to(device), masks.to(device))
                optimizer.zero_grad()
                composed_loss, projection_loss, attribute_loss_value = criterion(projection, attribute, vec_labels.to(device), attribute_tensor, no_relation_tensor)
                composed_loss.backward() # questo aggiornamento tocca TUTTI i pesi dei layer dedicati sia nel caso di no_relation  che nell'altro?
                optimizer.step()
                train_epoch_loss += composed_loss
                train_projection_loss += projection_loss
                train_attribute_loss += attribute_loss_value
            
            self.eval()
            if (e + 1) % 2 == 0:
                val_epoch_loss = 0
                val_projection_loss, val_attribute_loss = 0, 0
                epoch_labels, epoch_predictions = tuple(), tuple()
                with torch.no_grad():
                    for sentences, masks, vec_labels, rel_labels, prop_labels, int_labels, no_relation_tensor, attribute_tensor in val_loader:
                        projection, attribute = self(sentences.to(device), masks.to(device))
                        composed_loss, projection_loss, attribute_loss_value = criterion(projection, attribute, vec_labels.to(device), attribute_tensor, no_relation_tensor)

                        epoch_labels += prop_labels
                        epoch_predictions += self.convert(projection, attribute, space, device)

                        val_epoch_loss += composed_loss
                        val_projection_loss += projection_loss
                        val_attribute_loss += attribute_loss_value


                self.print_metrics(epoch_labels, epoch_predictions)
                train_loss = train_epoch_loss/len(train_loader)
                train_projection = train_projection_loss/len(train_loader)
                train_attribute = train_attribute_loss/len(train_loader)
                val_loss = val_epoch_loss/len(val_loader) 
                val_projection = val_projection_loss/len(val_loader) 
                val_attribute = val_attribute_loss/len(val_loader) 

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                    "Train_Loss: {:.3f}.. ".format(train_loss),
                    "Train_projection_Loss: {:.3f}.. ".format(train_projection),
                    "Train_attribute_Loss: {:.3f}.. ".format(train_attribute),
                    "Val_Loss: {:.3f}.. ".format(val_loss),
                    "Val_projection_Loss: {:.3f}.. ".format(val_projection),
                    "Val_attribute_Loss: {:.3f}.. ".format(val_attribute))
                    



    def convert(self, projections, attributes, space, device):
        cos = CosineSimilarity(dim=1)
        # questo pezzo potrebbe essere ricevuto direttamente ceom parametro così nn deve essere computato per ogni batch
        ps_properties = torch.Tensor()
        for index, row in space.iterrows():
            label_vec = torch.tensor([space.loc[index]])
            ps_properties = torch.cat([ps_properties, label_vec], dim=0)
        
        predictions = tuple()
        for projection, attribute in zip(projections, attributes) :
            if attribute < 0.5:
                projection = projection.reshape(1,-1)
                sims = cos(projection.to(device), ps_properties.to(device))
                arg_max = torch.argmax(sims).item()
                predictions += (space.index[arg_max],)
                if space.index[arg_max] == 'no_relation':
                    print('unexpected no_relation')
            else:
                predictions += ("no_relation",)

        return predictions



    def print_metrics(self, labels:list, predictions):
      #  print("ground truth labels: {}".format(set(labels)))
      #  print("prediction labels: {}".format(set(predictions)))
        print(metrics.classification_report(labels, predictions, labels=list(set(labels))))
        print("accuracy: {}".format(metrics.accuracy_score(labels, predictions)))

     #   corrects =0
     #   for label, prediction in zip(labels, predictions):
     #       if label==prediction:
     #           corrects += 1
     #   print("rechecked accuracy: {}".format(corrects/len(labels)))  






  #  def calculate_corrects(self, outputs, space, vec_labels, device):
  #      cos = CosineSimilarity(dim=0)
  #      batch_corrects = 0
  #      for i in range(vec_labels.shape[0]):
  #          max_sim=0
  #          closest_candidate=None
  #          for property, row_space in space.iterrows():
  #              vector = torch.tensor(space.loc[property]).to(device) 
  #              sim = cos(outputs[i], vector).item()
  #              if sim >= max_sim:
  #                  closest_candidate = vector
  #                  max_sim = sim
  #          if torch.equal(vec_labels[i], closest_candidate):
  #              batch_corrects += 1
  #      return batch_corrects

 #esperimento: anzichè confrontarlo solo con i vettori die 20 predicati che ti interssano, confronta con tutti, e trova a chi si avvicina di più