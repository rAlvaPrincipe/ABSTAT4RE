import pandas as pd
from pandas.core.frame import DataFrame
from sparql import Sparql as sparql

class Profile:

    # takes as input the directory path with profiles, the name of the statistics to consider (frequency, instances) a boolean clean
    def __init__(self, profile_dir, statistic, clean, artificial_props):
        cols = ["subject","predicate","object", statistic]
        if(statistic == "instances"):
            obj_akp = pd.read_csv(profile_dir + "/object_patterns.txt", sep=";", names=cols)
            dt_akp = pd.read_csv(profile_dir + "./datatype_patterns.txt", sep=";", names=cols)
        else:
            obj_akp = pd.read_csv(profile_dir + "/object_akp.txt", sep=";", names=cols)
            dt_akp = pd.read_csv(profile_dir + "/datatype_akp.txt", sep=";", names=cols)

        self.profile =  obj_akp.append(dt_akp)
        self.profile = self.profile.astype({"subject": str, "predicate": str, "object": str})
        self.profile["pair"]= self.profile["subject"]+"#-#"+self.profile["object"]
        if clean:
            self.clean()

        if artificial_props:
            self.profile = self.profile.append(self.build_artificial_properties())

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
        
    #  patterns filtering based on exact match of a list of properties or contains on keywords
    def filter(self, profile: DataFrame, position,  keywords, exact):
        if exact: 
            sub_df = profile[profile[position].isin(keywords)]
        else:
            or_list = " ".join(keywords).replace(" ", "|")
            sub_df = profile[profile[position].str.contains(or_list, case=False)]

        return sub_df
        

    #  takes a set of keywords and return all /preds/subjs/objs that contain that keywords
    def approximated_match(self, kewords, position):
        temp =  self.filter(self.profile, position, kewords, False)
        constraints = list(temp.predicate.unique())
        # extend with equivalent properties 
        eq_constraints = sparql.equivalentProperty(constraints)
        constraints.extend(eq_constraints)
        return constraints


    # takes a set of types and return all dbpedia subtypes enriched with equvalentClass concepts
    def subtypes_match(self, types):
        constraints = sparql.get_subTypes(types)
        constraints.extend(sparql.equivalentClasses(constraints))
        return constraints


    def build_artificial_property(self, label, preds_constraints, subj_constraints, obj_constraints):
        temp =  self.filter(self.profile, "predicate", preds_constraints, True)
        if len(obj_constraints) > 0:
            temp = self.filter(temp, "object", obj_constraints, True).sort_values(by=["frequency"], ascending=False)
        if len(subj_constraints) > 0:
            temp = self.filter(temp, "subject", subj_constraints, True).sort_values(by=["frequency"], ascending=False)

        temp["predicate"] = label
        temp  = temp.groupby(["subject", "predicate", "object", "pair"])["frequency"].sum().reset_index()
        return temp.sort_values(by=["frequency"], ascending=False)


    def build_artificial_properties(self):
        artifical_props_df = DataFrame()
        preds_constraints = self.approximated_match(["headquarter"], "predicate")
        subj_constraints = self.subtypes_match(["http://dbpedia.org/ontology/Organisation"])
        obj_constraints = ['http://wikidata.dbpedia.org/ontology/City', 'http://schema.org/City', 'http://wikidata.dbpedia.org/ontology/City', 'http://www.wikidata.org/entity/Q515']
        city_of_headquarters = self.build_artificial_property("org:city_of_headquarters", preds_constraints, subj_constraints, obj_constraints)
        print(city_of_headquarters)
        artifical_props_df  = artifical_props_df.append(city_of_headquarters)
        
        obj_constraints = ["http://www.wikidata.org/entity/Q6256", "http://dbpedia.org/ontology/Country", "http://schema.org/Country", "http://wikidata.dbpedia.org/ontology/Country"]
        country_of_headquarters = self.build_artificial_property("org:country_of_headquarters", preds_constraints, subj_constraints, obj_constraints)
        print(country_of_headquarters)
        artifical_props_df  = artifical_props_df.append(country_of_headquarters)

        obj_constraints = ["http://dbpedia.org/ontology/AdministrativeRegion", "http://schema.org/AdministrativeArea", "http://www.wikidata.org/entity/Q3455524"]
        stateorprovince_of_headquarters = self.build_artificial_property("org:stateorprovince_of_headquarters", preds_constraints, subj_constraints, obj_constraints)
        print(stateorprovince_of_headquarters)
        artifical_props_df  = artifical_props_df.append(stateorprovince_of_headquarters)


        preds_constraints = self.approximated_match(["residence"], "predicate")
        subj_constraints = self.subtypes_match(["http://dbpedia.org/ontology/Person"])
        obj_constraints = ['http://wikidata.dbpedia.org/ontology/City', 'http://schema.org/City', 'http://wikidata.dbpedia.org/ontology/City', 'http://www.wikidata.org/entity/Q515']
        cities_of_residence = self.build_artificial_property("per:cities_of_residence", preds_constraints, subj_constraints, obj_constraints)
        print(cities_of_residence)
        artifical_props_df  = artifical_props_df.append(cities_of_residence)

        obj_constraints = ["http://www.wikidata.org/entity/Q6256", "http://dbpedia.org/ontology/Country", "http://schema.org/Country", "http://wikidata.dbpedia.org/ontology/Country"]
        countries_of_residence = self.build_artificial_property("per:countries_of_residence", preds_constraints, subj_constraints, obj_constraints)
        print(countries_of_residence)
        artifical_props_df  = artifical_props_df.append(countries_of_residence)

        obj_constraints = ["http://dbpedia.org/ontology/AdministrativeRegion", "http://schema.org/AdministrativeArea", "http://www.wikidata.org/entity/Q3455524"]
        stateorprovinces_of_residence = self.build_artificial_property("per:stateorprovinces_of_residence", preds_constraints, subj_constraints, obj_constraints)
        print(stateorprovinces_of_residence)
        artifical_props_df  = artifical_props_df.append(stateorprovinces_of_residence)


        preds_constraints = self.approximated_match(["alias", "alternatename"], "predicate")
        subj_constraints = self.subtypes_match(["http://dbpedia.org/ontology/Organisation"])
        obj_constraints = []
        org_alternates_names = self.build_artificial_property("org:alternate_names", preds_constraints, subj_constraints, obj_constraints)
        print(org_alternates_names)
        artifical_props_df  = artifical_props_df.append(org_alternates_names)

        subj_constraints = self.subtypes_match(["http://dbpedia.org/ontology/Person"])
        per_alternates_names = self.build_artificial_property("per:alternate_names", preds_constraints, subj_constraints, obj_constraints)
        print(per_alternates_names)
        artifical_props_df  = artifical_props_df.append(per_alternates_names)


        preds_constraints = self.approximated_match(["employee", "member"], "predicate")
        subj_constraints =  self.subtypes_match(["http://dbpedia.org/ontology/Organisation"])
        obj_constraints =  self.subtypes_match(["http://dbpedia.org/ontology/Organisation"])
        org_members = self.build_artificial_property("org:members", preds_constraints, subj_constraints, obj_constraints)
        print(org_members)
        artifical_props_df  = artifical_props_df.append(org_members)

        subj_constraints = self.subtypes_match(["http://dbpedia.org/ontology/Person"])
        obj_constraints =  self.subtypes_match(["http://dbpedia.org/ontology/Organisation"])
        per_employee_of = self.build_artificial_property("per:employee_of", preds_constraints, subj_constraints, obj_constraints)
        print(per_employee_of)
        artifical_props_df  = artifical_props_df.append(per_employee_of)

        subj_constraints =  self.subtypes_match(["http://dbpedia.org/ontology/Organisation"])
        obj_constraints =  self.subtypes_match(["http://dbpedia.org/ontology/Person"])
        org_top_members_employees = self.build_artificial_property("org:top_members_employees", preds_constraints, subj_constraints, obj_constraints)
        print(org_top_members_employees)
        artifical_props_df  = artifical_props_df.append(org_top_members_employees)

        return artifical_props_df
