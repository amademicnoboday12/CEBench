from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Qdrant
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient, models
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from core.utils.private_embedding import embeddings
from core.monitor.timer_callback import TimerCallback
from core.metrics.post_processing import post_processing_mental


def query(
    query_text,
    llm_model_name,
    template_path,
    context="",
    rag_collection=None,
    top_k=10,
    quantization=None,
):
    client = QdrantClient(url="localhost:6333")
    llm = Ollama(model=llm_model_name)
    time_rag = {}
    time_llm = [0]
    handler = TimerCallback(time_llm, time_rag)
    template_doc = TextLoader(template_path).load()[0].page_content
    rag_prompt = PromptTemplate.from_template(template_doc)

    if rag_collection is None:
        rag_chain = rag_prompt | llm | StrOutputParser()
        answer = rag_chain.invoke(
            {"context": context, "input": query_text}, config={"callbacks": [handler]}
        )
        return answer.replace("\n", " <br> "), sum(time_rag.values()), time_llm[0], handler.input_token, handler.output_token

    quantization_param = {
        "search_params": models.SearchParams(
            quantization=models.QuantizationSearchParams(
                ignore=False,
                rescore=True,
                oversampling=2.0,
            )
        )
    }
    if quantization is None:
        quantization_param = {}
    if not client.collection_exists(rag_collection):
        raise ValueError(f"Collection {rag_collection} does not exist")
    doc_store = Qdrant(
        client=client, collection_name=rag_collection, embeddings=embeddings
    )

    def format_docs(docs):
        limit = min(top_k, len(docs))
        return "\n\n".join(docs[i].page_content for i in range(limit))

    retriever = doc_store.as_retriever(search_kwargs=quantization_param)

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(query_text, config={"callbacks": [handler]})

    return post_processing_mental(answer.replace("\n", " <br> ")), sum(time_rag.values()), time_llm[0],handler.input_token,handler.output_token
