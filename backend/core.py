import os
from typing import Any, List, Dict
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
# from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
# from langchain.callbacks import StreamingStdOutCallbackHandler
import pinecone

from consts import INDEX_NAME


pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0)
    doc_array = docsearch.similarity_search(query, k=10)
    # handler = StreamingStdOutCallbackHandler()
    sources = [doc.metadata['source'] for doc in doc_array]
    # for i, n in enumerate(doc_array):
    #     print(doc_array[i]['metadata']['source'])
    


    # qa = ConversationalRetrievalChain.from_llm(llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True, callbacks=[handler])
    qa = ConversationalRetrievalChain.from_llm(llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True)
    print("fetching answer!")
    return qa({"question": query, "chat_history":chat_history}), sources


def summary_llm(query:str, chat_history: List[Dict[str,Any]] = [])-> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0)
    
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
    
    compressor = LLMChainExtractor.from_llm(chat)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=docsearch.as_retriever(search_type = "mmr")
    )
    # retriever = docsearch.as_retriever(search_type = "mmr", search_kwargs={"k": 4})
    # retriever = docsearch.max_marginal_relevance_search(query=q, k=2, fetch_k=3)
        
    from langchain.memory import ConversationTokenBufferMemory
    memory = ConversationTokenBufferMemory(llm=chat,memory_key="chat_history", return_messages=True,input_key='question',max_token_limit=1000)
    from langchain.prompts.prompt import PromptTemplate
    _template =  """Write a concise summary of the following:{question}
    Context: 
    {chat_history}
    CONCISE SUMMARY:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
    # CONDENSE_QUESTION_PROMPT = PromptTemplate(input_variables=['question'], template=_template)

    spark = """
    "Your job is to produce a final summary\n"
    "We have provided an existing summary up to a certain point, including sources: {existing_answer}\n"
    "We have the opportunity to refine the existing summary"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_str}\n"
    "------------\n"
    "If you do update it, please update the sources as well. "
    "If the context isn't useful, return the original summary."
    """
    from langchain.prompts import (
        ChatPromptTemplate,
        PromptTemplate,
        SystemMessagePromptTemplate,
        AIMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(spark)
    # instruction = HumanMessagePromptTemplate.from_template(prompt_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

    from langchain.chains import ConversationalRetrievalChain, LLMChain
    from langchain.chains.qa_with_sources import load_qa_with_sources_chain
    from langchain.schema import (
        AIMessage,
        HumanMessage,
        SystemMessage
    )

    question_generator = LLMChain(llm=chat, prompt=CONDENSE_QUESTION_PROMPT, verbose=True)
    answer_chain = load_qa_with_sources_chain(chat, chain_type="refine", verbose=True,question_prompt= CONDENSE_QUESTION_PROMPT, refine_prompt=chat_prompt)
    # from langchain.chains.summarize import load_summarize_chain
    
    # chain = load_summarize_chain(
    #   llm=llm,
    #   chain_type="refine",
    #   question_prompt=prompt,
    #   refine_prompt=refine_prompt,
    #   return_intermediate_steps=True,
    #   input_key="input_documents",
    #   output_key="output_text",
    # )
    
    chain = ConversationalRetrievalChain(
                retriever=compression_retriever,
                question_generator=question_generator,
                combine_docs_chain=answer_chain,
                verbose=True,
                memory=memory,
                rephrase_question=False
    )
    print("fetching answer summary!")
    return chain({"question": query})

    # qa = ConversationalRetrievalChain.from_llm(llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True)
    # print("fetching answer!")
    # return qa({"question": query, "chat_history":chat_history})



if __name__ == "__main__":
    run_llm(query="What are the benefits of Juniper BNG CUPS?")

# # core.py
# import os
# from typing import Any, List, Dict
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import ConversationalRetrievalChain
# from langchain.vectorstores import Pinecone
# from langchain.callbacks import StreamingStdOutCallbackHandler
# import pinecone

# from consts import INDEX_NAME


# pinecone.init(
#     api_key=os.environ["PINECONE_API_KEY"],
#     environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
# )


# async def run_llm(query: str, chat_history: List[Dict[str, Any]] = []) -> Any:
#     embeddings = OpenAIEmbeddings()

#     docsearch = Pinecone.from_existing_index(
#         index_name=INDEX_NAME, embedding=embeddings
#     )

#     chat = ChatOpenAI(verbose=True, temperature=0)

#     doc_array = docsearch.similarity_search(query, k=10)
#     sources = [doc.metadata['source'] for doc in doc_array]

#     qa = ConversationalRetrievalChain.from_llm(
#         llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
#     )

#     async def stream_response():
#         while True:
#             response = await qa.next_response()
#             st.chat_message("Bard", response)

#     asyncio.get_event_loop().create_task(stream_response())

#     return qa({"question": query, "chat_history": chat_history}), sources


# if __name__ == "__main__":
#     asyncio.run(run_llm(query="What are the benefits of Juniper BNG CUPS?"))

