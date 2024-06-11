import yaml
from yaml import Loader
import os
from query_llms import query
from load_rag import load_knowledge_base
from loguru import logger
import time
from scorer import scorer_mental
import nvidia_smi
import plan_recommender


current_time=time.time()

rag_log_file = f"results/evaluation_with_rag.log_{current_time}"
response_with_rag_log_file = f"results/response_with_rag.txt_{current_time}"

context_log_file = f"loresultsgs/evaluation_with_context.log_{current_time}"
response_with_context_log_file = f"results/response_with_context.txt_{current_time}"

metric_log_file = f"results/metric.log_{current_time}"

configures = yaml.load(open("experiment_config.yml").read(), Loader=Loader)
TEMPLATE_PATH = f'./templates/{configures["template_file_name"]}'
KNOWLEDGE_PATH = f'./knowledge_base/{configures["knowledge_base_dir"]}'
MODELS = configures["models"]
RAG_COLLECTION = configures["rag_collection_name"]
QUANTIZATION = configures["quantization"]
CHUNK_SIZE = configures["chunk_size"]
TOP_K = configures["top_k"]
QUERY_FILE = f'./queries/{configures["query_file_name"]}_query.txt'
CONTEXT_FILE = f'./queries/{configures["query_file_name"]}_context.txt'
GROUND_TRUTH= f'./queries/{configures["query_file_name"]}_ground_truth.txt'


# Please complete this function to read in the queries from the file
# return query_text_list, context_list_for_each_query
# if no context, then return return query_text_list, None
def read_in_queries():
    with open(QUERY_FILE) as f:
        query_text_list = f.readlines()

    return zip(
        list(range(len(query_text_list))), query_text_list, [""] * len(query_text_list)
    )

def read_in_gd():
    with open(GROUND_TRUTH) as f:
        ground_truth_list = f.readlines()
    return ground_truth_list


if __name__ == "__main__":
    from query_llms import query
    from load_rag import load_knowledge_base
    context_enable = os.path.exists(CONTEXT_FILE)
    exp_id = 0
    if RAG_COLLECTION is not None:
        with open(metric_log_file, "a") as m:
            m.write("Experiment ID,Score,Max GPU Memory\n")
        with open(response_with_rag_log_file, "w") as f:
            logger.add(rag_log_file, format="{message}", level="INFO")
            logger.info("Experiment ID,Query ID,Collection,Quantization,Chunk,Model,Top K,Input token,Output token,Time RAG,Time LLM")
            for qt in QUANTIZATION:
                for chunk in CHUNK_SIZE:
                    load_knowledge_base(RAG_COLLECTION, KNOWLEDGE_PATH, qt, chunk)
                    for model in MODELS:
                        for top_k in TOP_K:
                            answers=[]
                            exp_id += 1
                            max_gpu_mem=0
                            for qid, query_text, context in read_in_queries():
                                answer, time_rag, time_llm, input_token,output_token = query(
                                    query_text,
                                    model,
                                    TEMPLATE_PATH,
                                    rag_collection=RAG_COLLECTION,
                                    top_k=top_k,
                                )
                                #headers: Experiment ID, Query ID, Collection, Quantization, Chunk, Model, Top K, Input token, Output token, Time RAG, Time LLM
                                answers.append(answer)
                                logger.info(
                                    f"{exp_id}, {qid}, {RAG_COLLECTION}, {qt},{chunk}, {model},{top_k}, {input_token}, {output_token}, {time_rag}, {time_llm}"
                                )
                                f.write(answer + "\n")
                                nvidia_smi.nvmlInit()
                                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
                                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                                if info.used > max_gpu_mem:
                                    max_gpu_mem=info.used
                                nvidia_smi.nvmlShutdown()
                            score=scorer_mental(read_in_gd(),answers)
                            with open(metric_log_file, "a") as m:
                                m.write(f"{exp_id}, {score},{max_gpu_mem/(1024**3)}\n")
        plan_recommender.plan_recommender('VM',metric_log_file,rag_log_file)
    else:
        with open(metric_log_file, "a") as m:
            m.write("Experiment ID,Score,Max GPU Memory\n")
        with open(response_with_context_log_file, "w") as f:
            logger.add(context_log_file, format="{message}", level="INFO")
            logger.info("Experiment ID,Query ID,Model,Input token,Output token,Time RAG,Time LLM")
            for model in MODELS:
                exp_id += 1
                answers=[]
                max_gpu_mem=0
                for qid, query_text, context in read_in_queries():
                    if context_enable:
                        answer, time_rag, time_llm = query(
                            query_text,
                            model,
                            TEMPLATE_PATH,
                            context,
                            rag_collection=RAG_COLLECTION,
                        )
                    else:
                        answer, time_rag, time_llm, input_token, output_token = query(
                            query_text,
                            model,
                            TEMPLATE_PATH,
                            rag_collection=RAG_COLLECTION,
                        )
                    #headers: Experiment ID, Query ID, Model, Input token, Output token, Time RAG, Time LLM
                    answers.append(answer)
                    logger.info(
                        f"{exp_id},{qid},  {model}, {input_token},{output_token}, {time_rag}, {time_llm}"
                    )
                    f.write(answer + "\n")
                    nvidia_smi.nvmlInit()
                    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
                    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                    if info.used > max_gpu_mem:
                        max_gpu_mem=info.used   
                    nvidia_smi.nvmlShutdown()
                score=scorer_mental(read_in_gd(),answers)
                with open(metric_log_file, "a") as m:
                    m.write(f"{exp_id}, {score},{max_gpu_mem/(1024**3)}\n")
        plan_recommender.plan_recommender('VM',metric_log_file,context_log_file)
    