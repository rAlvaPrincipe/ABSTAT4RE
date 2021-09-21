import pandas as pd
import json


subspace = pd.read_csv("/home/ralvaprincipe/ABSTAT4RE/data/outputs/prop_spaces/PS_kbp37_nonZeroDims.csv", index_col=0)
with open("/home/ralvaprincipe/ABSTAT4RE/metadata/kbp37_mapping.json") as json_file:
    mappings = json.load(json_file)

count = 0
to_drop = set()
for key in mappings.keys():
    prop = mappings[key]
    if "@" in prop:
        count += 1
        orignal_prop = prop[:prop.find("@")-1]
        # add a column for the additional dimension
        dim_name = "add"+ str(count)
        subspace[dim_name] = 0
        # the artificial property is basically a copy of the original one
        subspace.loc[prop]=subspace.loc[orignal_prop]
        subspace.loc[prop, dim_name] = 1
        to_drop.add(orignal_prop)

subspace.drop(to_drop, axis=0, inplace=True)
print(subspace)
