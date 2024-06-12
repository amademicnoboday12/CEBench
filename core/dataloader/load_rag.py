from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from core.utils.private_embedding import embeddings


def load_knowledge_base(rag_collection, doc_path, quantization=None, chunk_size=512):
    overlap = chunk_size // 8
    client = QdrantClient()

    if quantization == "pq":
        quantization_conf = models.ProductQuantization(
            product=models.ProductQuantizationConfig(
                compression=models.CompressionRatio.X4, always_ram=True
            )
        )
    elif quantization == "scalar":
        quantization_conf = models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8, always_ram=True
            )
        )
    else:
        quantization_conf = None
    loader = DirectoryLoader(
        doc_path,
        glob="**/*.txt",
        show_progress=True,
        use_multithreading=True,
        loader_cls=TextLoader,
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n"]
    )
    flatmap = lambda f, xs: [y for ys in xs for y in f(ys)]
    chunks = flatmap(text_splitter.split_documents, documents)
    dimensions = len(embeddings.embed_query("this is a test"))
    client.delete_collection(collection_name=rag_collection)
    client.create_collection(
        collection_name=rag_collection,
        vectors_config=VectorParams(size=dimensions, distance=Distance.COSINE),
        quantization_config=quantization_conf,
    )
    qdrant = Qdrant(
        client=client, embeddings=embeddings, collection_name=rag_collection
    )
    qdrant.add_documents(chunks)
    print(
        f"Added {len(chunks)} chunks, {len(documents)} documents to collection {rag_collection}"
    )
