from langchain import RagRetriever, RagSequenceChain

def test_rag_integration():
    retriever = RagRetriever.from_pretrained("facebook/rag-token-nq")
    chain = RagSequenceChain(retriever=retriever)

    query = "What is the capital of France?"
    response = chain.run(query)

    assert response is not None
    assert "Paris" in response