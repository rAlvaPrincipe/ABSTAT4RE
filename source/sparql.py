from SPARQLWrapper import SPARQLWrapper, JSON
import time 

class Sparql:

    @staticmethod
    def enpoint_query(query, endpoint):
        if endpoint == "dbpedia":
            sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        else:
            sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        sparql.setQuery(query) 
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()  
        return results


    #given a set of properties, return the set of equivalent properties
    @staticmethod
    def equivalentProperty(properties):
        eq_props = set()
        for property in properties:
            query = """ SELECT ?equivalent_prop
                        WHERE { <""" + property + """> owl:equivalentProperty ?equivalent_prop.
                        } """
                        
            results = Sparql.enpoint_query(query, "dbpedia")
            for result in results["results"]["bindings"]:
                eq_props.add(result["equivalent_prop"]["value"]) 

        return eq_props

    #given a set of classes, return the set of equivalent classes
    @staticmethod
    def equivalentClasses(classes):
        eq_classes = set()
        for concept in classes:
            query = """ SELECT ?equivalent_class
                        WHERE { <""" + concept + """> owl:equivalentClass ?equivalent_class.
                        } """
                        
            results = Sparql.enpoint_query(query, "dbpedia")
            for result in results["results"]["bindings"]:
                eq_classes.add(result["equivalent_class"]["value"]) 

        return eq_classes

    #given a type, returns the set of subtypes applying transitive closure  of subClassOf and including the input type
    @staticmethod
    def get_subTypes(types):
        subtypes = list()
        for type in types:
            query = """ SELECT ?subtype 
                        WHERE { ?subtype rdfs:subClassOf*  <""" + type + """>
                        } """
            results = Sparql.enpoint_query(query, "dbpedia")
            for result in results["results"]["bindings"]:
                    subtypes.append(result["subtype"]["value"]) 
        return subtypes
        


    # take a list of resources and returns a list of pairs resource --> label
    @staticmethod
    def getLabels(resources):
        res_labels = list()
        for source in resources:
            query = """ SELECT ?label
                        WHERE { <""" + source + """> rdfs:label ?label .
                        FILTER (langMatches( lang(?label), "EN" ) )
                        } """

            print(source)
            if "wikidata.org" in source.lower():
                time.sleep(3)
                results = Sparql.enpoint_query(query, "wikidata")
            else:
                results = Sparql.enpoint_query(query, "dbpedia")

            if len(results["results"]["bindings"]) > 0:
                label = results["results"]["bindings"][0]["label"]["value"]
                res_labels.append((source, label))
            else: 
                res_labels.append((source, "NULL"))

        return res_labels


#print(Sparql.get_subTypes(["http://dbpedia.org/ontology/Organisation"]))