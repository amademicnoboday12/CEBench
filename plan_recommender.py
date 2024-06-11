from yaml import Loader
import yaml
import pandas as pd
from paretoset import paretoset
def plan_recommender(resource_type,metric_log_file,evaluation_file):
    configures = yaml.load(open("configs/instance_quote.yml").read(), Loader=Loader)
    resources=configures["resources"]
    for resource in resources:
        if resource["type"]==resource_type:
            instances=resource["instances"]
            quote=[]
            for i in instances:
                quote.append([i["name"],float(i['TFlops'])/77.97,float(i['memory']),float(i['price'])])
            
    data=pd.DataFrame(quote,columns=["Instance","TFlops","Memory","Price"])
    #headers = ['Experiment ID','Score', ' Max GPU Memory']
    metric=pd.read_csv(metric_log_file,index_col=False)
    eval=pd.read_csv(evaluation_file,index_col=False)
    eval=eval[['Experiment ID','Quantization','Chunk','Model','Top K','Time LLM']].groupby(['Experiment ID','Quantization','Chunk','Model','Top K']).mean()
    eval=eval.reset_index()
    eval=eval.merge(metric,on=['Experiment ID'])
    all_results=eval.merge(data,how='cross')
    valids=all_results[all_results['Max GPU Memory']<all_results['Memory']]
    valids['Est Time']=valids['Time LLM']/valids['TFlops']
    valids['Est Cost']=valids['Est Time']*valids['Price']*1000/3600
    pareto_mask=paretoset(valids[['Score','Est Cost']],sense=['min','min'])
    valids[pareto_mask].to_csv(f'{evaluation_file}_pareto.csv',index=False)
