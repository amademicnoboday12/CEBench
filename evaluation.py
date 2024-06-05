import yaml
from yaml import Loader
import os
from query_llms import query
from load_rag import load_knowledge_base
from loguru import logger
import time

current_time=time.time()

rag_log_file = f"logs/evaluation_with_rag.log_{current_time}"
response_with_rag_log_file = f"logs/response_with_rag.txt_{current_time}"

context_log_file = f"logs/evaluation_with_context.log_{current_time}"
response_with_context_log_file = f"logs/response_with_context.txt_{current_time}"

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


# Please complete this function to read in the queries from the file
# return query_text_list, context_list_for_each_query
# if no context, then return return query_text_list, None
def read_in_queries():
    with open(QUERY_FILE) as f:
        query_text_list = f.readlines()
    return zip(
        list(range(len(query_text_list))), query_text_list, [""] * len(query_text_list)
    )


if __name__ == "__main__":
    from query_llms import query
    from load_rag import load_knowledge_base

    context_enable = os.path.exists(CONTEXT_FILE)
    exp_id = 0
    if RAG_COLLECTION is not None:
        with open(response_with_rag_log_file, "w") as f:
            logger.add(rag_log_file, format="{message}", level="INFO")
            for qt in QUANTIZATION:
                for chunk in CHUNK_SIZE:
                    load_knowledge_base(RAG_COLLECTION, KNOWLEDGE_PATH, qt, chunk)
                    for model in MODELS:
                        for top_k in TOP_K:
                            for qid, query_text, context in read_in_queries():
                                exp_id += 1
                                answer, time_rag, time_llm, input_token,output_token = query(
                                    query_text,
                                    model,
                                    TEMPLATE_PATH,
                                    rag_collection=RAG_COLLECTION,
                                    top_k=top_k,
                                )
                                logger.info(
                                    f"Experiment ID: {exp_id}, Query ID:{qid}, Collection: {RAG_COLLECTION}, Quantization: {qt},Chunk: {chunk} Model: {model}, Top K: {top_k}, Input token:{input_token}, Output token:{output_token}, Time RAG: {time_rag}, Time LLM: {time_llm}"
                                )
                                f.write(answer + "\n")
    else:
        with open(response_with_context_log_file, "w") as f:
            logger.add(context_log_file, format="{message}", level="INFO")
            for model in MODELS:
                for qid, query_text, context in read_in_queries():
                    exp_id += 1
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
                    logger.info(
                        f"Experiment ID: {exp_id}, Query ID:{qid}, Model: {model}, Input token:{input_token}, Output token:{output_token}, Time RAG: {time_rag}, Time LLM: {time_llm}"
                    )
                    f.write(answer + "\n")
