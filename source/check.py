import torch
from torch.nn import CosineSimilarity
from profile import Profile
class Check:


    def check_vec_labels_correcteness(dataset, space, props_mapping):
        vec_labels = torch.load("/home/ralvaprincipe/ABSTAT4RE/data/outputs/vec_labels/labels-vec_kbp37_val_dbp2016-full-onlyNonZeroDims.pt")
        
        count =0
        for i, row in dataset.iterrows():
            index = props_mapping[row["rel"]]
            label = torch.tensor(space.loc[index])
            if(row["rel"] == "no_relation"):
                count += 1
            if torch.equal(label, vec_labels[i]) == False:
                print("hey")

        print("end")
        print(count)


    def check2(val_loader, space):
        for sentences, masks, vec_labels, rel_labels, prop_labels in val_loader:
            Check.convert(vec_labels, space, prop_labels)
            #for rel, prop in zip(rel_labels, prop_labels):
            #    print((rel, prop ))
        print("end")

    def convert(outputs, space, prop_labels):
        cos = CosineSimilarity(dim=0)
        closest_candidate=None
        for output, prop_label in zip(outputs, prop_labels):
            max_sim = -1
            for property, row_space in space.iterrows():
                vector = torch.tensor(space.loc[property])
                sim = cos(output, vector).item()
                if sim >= max_sim:
                    closest_candidate = property
                    max_sim = sim
        
            if prop_label != closest_candidate:
                print(torch.sum(output))
                print((prop_label, closest_candidate, max_sim ))


    def check_PS(space):
        temp = space.copy()
        temp['sum'] = temp.sum(axis=1)
        print(temp["sum"])
        print(space)


    def check_profile(profile):
        print(profile)
        print(profile[profile["predicate"]=="http://dbpedia.org/ontology/nationality"])
        print(profile[profile["predicate"]=="http://dbpedia.org/ontology/formationDate"])
        print(profile[profile["predicate"]=="http://dbpedia.org/ontology/title"])
