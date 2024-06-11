from yaml import Loader
import yaml
def plan_recommender(resource_type):
    configures = yaml.load(open("instance quote.yml").read(), Loader=Loader)
    resources=configures["resources"]
    for resource in resources:
        if resource["type"]==resource_type:
            instances=resource["instances"]
    

