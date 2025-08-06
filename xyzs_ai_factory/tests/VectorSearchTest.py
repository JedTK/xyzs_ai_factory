from xyzs_ai_factory.Module.BaseFactory.VectorSearch import VectorSearchFactory
from xyzs_ai_factory.Module.VectorSearch import FaissVectorSearch

vector_factory = VectorSearchFactory()
vector_factory.register("faiss", FaissVectorSearch)
vector_factory.init_instance(name="faiss", config={
    "model_path": "/Users/jedwong/Work/JProject/AI/model/bge-small-zh-v1.5",
    "faiss_file": "/Users/jedwong/Work/chaineff/AssistHubWork/xyzs_ai_factory/resources/FAQ.faiss",
})

distances, indices = vector_factory.get_instance("faiss").search(context={}, params={
    "query": "请问怎么充值？",
    "top_k": 5,
})
print(distances)
print(indices)
