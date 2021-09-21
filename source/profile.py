import pandas as pd


class Profile:

    # takes as input the directory path with profiles, the name of the statistics to consider (frequency, instances) a boolean clean
    def __init__(self, profile_dir, statistic, clean):
        cols = ["subject","predicate","object", statistic]
        if(statistic == "instances"):
            obj_akp = pd.read_csv(profile_dir + "/object_patterns.txt", sep=";", names=cols)
            dt_akp = pd.read_csv(profile_dir + "./datatype_patterns.txt", sep=";", names=cols)
        else:
            obj_akp = pd.read_csv(profile_dir + "/object_akp.txt", sep=";", names=cols)
            dt_akp = pd.read_csv(profile_dir + "/datatype_akp.txt", sep=";", names=cols)

        self.profile =  obj_akp.append(dt_akp)
        self.profile["pair"]= self.profile["subject"]+"#-#"+self.profile["object"]
        if clean:
            self.clean()


    def clean(self):
        self.drop_wikiprops_patterns()
        self.drop_rare_pairs(1)
        self.drop_infrequent_patterns(10)
        self.drop_small_patternset_props(1)


    # removes patterns which predicate contains "wiki" 
    def drop_wikiprops_patterns(self):
        self.profile = self.profile[~(self.profile.predicate.str.contains("wiki", case=False))]


    # remove rare pairs (pairs which occur less or equals than n times)
    def drop_rare_pairs(self, n):
        temp = self.profile.groupby(["subject", "object"]).size().to_frame("size").reset_index()
        temp = temp[temp["size"] <= n]
        temp["pair"]=  temp["subject"]+"#-#"+ temp["object"]
        self.profile = self.profile[~self.profile["pair"].isin(temp["pair"].to_list())]


    # remove paterns with frequency <= freq
    def drop_infrequent_patterns(self, freq):
        self.profile = self.profile[self.profile["frequency"] > freq]

    
    # remove patterns for properties which have less or equals than n associated patterns
    def drop_small_patternset_props(self, n):
        temp = self.profile.groupby("predicate").size().to_frame("size").reset_index()
        temp = temp[temp["size"] <= n]
        self.profile = self.profile[~self.profile["predicate"].isin(temp["predicate"])]


    def summary(self):
        # pattern con frequenza 1
        t_2 = self.profile[self.profile["frequency"]<2]
        # predicati con un solo pattern
        t_3 = self.profile.groupby("predicate").size().to_frame("size").reset_index()
        t_3 =t_3[t_3["size"]<2]
        # pattern con frequenza=1 e 1 pattern
        t_4 = t_2.groupby("predicate").size().to_frame("size").reset_index()
        t_4 = t_4[t_4["size"]<2]
        # pairs che compaiono 1 sola volta
        t_5 = self.profile.groupby(["subject", "object"]).size().to_frame("size").reset_index()
        t_5 = t_5[t_5["size"]<2]
        predicates = self.profile["predicate"].drop_duplicates().tolist()

        print('{}: patterns in totale'.format(self.profile.shape[0]))
        print('{}: patterns con freq 1 \n'.format(t_2.shape[0]))
        print('{}: predicati nel dataset'.format( len(predicates)))
        print('{}: predicati con un solo pattern \n'.format(t_3.shape[0]))
        print('{}: predicati con un solo pattern e che hanno freq 1'.format(t_4.shape[0]))
        print('{}: pairs che non si ripetono nel profilo'.format(t_5.shape[0]))


    def df(self):
        return self.profile