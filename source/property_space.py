from profile import Profile
import pandas as pd
import numpy as np


class PropertySpace:

    # takes as input a profile or a dump path
    def __init__(self, *args):
        if(isinstance(args[0], Profile)):
            self.profile =  args[0].df()
            self.dimensions = pd.DataFrame()
            self.space = pd.DataFrame()
            self.buid_space()
        else:
            self.read(args[0])


    # TO-DO renderlo compatibile con "isntances"
    def probs(self):
        # addd column tot: for each predicate it contains the sum of associated frequencies
        sum_df = self.profile.groupby("predicate")["frequency"].sum().reset_index(name='tot')
        self.profile = pd.merge(self.profile, sum_df, how="inner", on="predicate")
        
        # calculate predicate probability for each pair <subject,object>
        self.profile['prob'] = self.profile.apply(lambda row: ((row.frequency * 100)/row.tot)*0.01, axis=1)


    def vectorize(self, temp):
        vec = np.zeros(self.dimensions.shape[0])
        for comp in temp["component"]:
            val = temp[temp["component"]==comp]["prob"].iloc[0]
            vec[comp]=val
        return(vec)


    def buid_space(self):
        self.probs() 
        # get axis labels and id for prodimensionspety space
        self.dimensions = self.profile[["pair"]].drop_duplicates().reset_index(drop="True")
        self.dimensions["component"] = self.dimensions.index
        self.dimensions.columns = ["component_label","component"]

        count=0
        predicates = self.profile["predicate"].drop_duplicates().tolist()
        #for pred in [ "http://dbpedia.org/ontology/name", "http://dbpedia.org/ontology/birthPlace"]:
        for pred in predicates:
            count += 1
            df_pred = self.profile.loc[self.profile["predicate"] == pred, ["pair", "prob"]]
            notzero_subset = self.dimensions[self.dimensions["component_label"].isin(df_pred["pair"])]
            temp = pd.merge(df_pred, notzero_subset, how="inner", left_on='pair', right_on='component_label').drop(columns=["pair","component_label"])
        
            vec = self.vectorize(temp)
            self.space[pred]=vec
            if(count%100==0):
                print(count)
        self.space = self.space.T


    def save(self, path):
        self.space.to_csv(path)


    def read(self, path):
        self.space = pd.read_csv(path, index_col=0)


    def df(self):
        return self.space

    
     # return only a df with the rows associates to the properties in the input list
    # if dim_reduc=True it find the dimensions which are 0 for each property and remove them
    def get_subspace(self, properties, dim_reduc, save, path):
        subspace = self.space.loc[properties]
        subspace.loc["no_relation"] = 0 

        if dim_reduc:
            # find and remove 0 columns (useless idmensions)
            zero_columns = list()
            for col in subspace.columns:
                is_zero = True
                for index, row in subspace.iterrows():
                    if row[col] != 0:
                        is_zero = False
                        break
                if is_zero:
                    zero_columns.append(col)     
                    
            subspace.drop(zero_columns, axis=1, inplace=True)
        if save:
            subspace.to_csv(path)

        return subspace



 #TO_DO: funzione che si riguarda il json dei mappngs, trova i cluster e aggiunge le componenti addizionali allo spazio originale e lo ritorna


#- input: AKPS
#- Data analysis funcs:
#         -  properties distribution
#         - summary
#- utils: 
#          -  properties simialirty
#          -  obtain a space with a subset of properties and optionally apply naive dimensioanl reduction