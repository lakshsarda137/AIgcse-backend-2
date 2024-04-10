from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from flask import Flask,jsonify,Request,request
from flask_cors import CORS
from langchain_google_vertexai import ChatVertexAI, HarmBlockThreshold, HarmCategory, HarmBlockThreshold, HarmCategory
from langchain_anthropic import ChatAnthropic

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.storage import LocalFileStore
from langchain_google_vertexai import ChatVertexAI, HarmBlockThreshold, HarmCategory, HarmBlockThreshold
chat = ChatVertexAI(project='chatbot-3793c', anthropic_version='vertex-2023-10-16', temperature=0.0,max_output_tokens=2040)
vertex=ChatVertexAI(project='chatbot-3793c',model_name='gemini-1.0-pro',temperature=0.0,max_output_tokens=2000)
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from llamaapi import LlamaAPI
from langchain_experimental.llms import ChatLlamaAPI
from langchain_experimental.chat_models import Llama2Chat
# Replace 'Your_API_Token' with your actual API token
llm_google = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    google_api_key='AIzaSyCW71zfRU69nAknLobdkOqjM1noPqqlxG0',
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,

    },
    temperature=0.45,
)
llama = LlamaAPI(api_token="LL-3KvWm5j6lg8d848cJFf1kAj0HobLwoS6Lx2R55yr1zafJc9KDsXDzkuOe7x19uhJ")
facebook=ChatLlamaAPI(client=llama, model_kwargs={"model_name":'llama-7b-chat',"temperature":0.0})
output_parser=StrOutputParser()
computer_science_all=DirectoryLoader('/Users/LakshSarda/PycharmProjects/pythonProject3/Computer_Science_Chatbot/CS_chatbot/question_papers_cs1_all',loader_cls=PyPDFLoader)
computer_science_all_documents=computer_science_all.load()
computer_science_loader_2023=DirectoryLoader('/Users/LakshSarda/PycharmProjects/pythonProject3/Computer_Science_Chatbot/CS_chatbot/question_papers_cs_1',loader_cls=PyPDFLoader)
computer_science_2023_documents=computer_science_loader_2023.load()
computer_science_loader_2021=DirectoryLoader('/Users/LakshSarda/PycharmProjects/pythonProject3/Computer_Science_Chatbot/CS_chatbot/question_papers_cs_2',loader_cls=PyPDFLoader)
computer_science_2021_documents=computer_science_loader_2021.load()
computer_science_loader_2019=DirectoryLoader('/Users/LakshSarda/PycharmProjects/pythonProject3/Computer_Science_Chatbot/CS_chatbot/question_papers_cs_3',loader_cls=PyPDFLoader)
computer_science_2019_documents=computer_science_loader_2019.load()
computer_science_loader_2016=DirectoryLoader('/Users/LakshSarda/PycharmProjects/pythonProject3/Computer_Science_Chatbot/CS_chatbot/question_papers_cs_4',loader_cls=PyPDFLoader)
computer_science_2016_documents=computer_science_loader_2016.load()
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=200,
    add_start_index=True
)
split_docs_all=text_splitter.split_documents(computer_science_all_documents)
split_docs=text_splitter.split_documents(computer_science_2023_documents)
split_docs_2021=text_splitter.split_documents(computer_science_2021_documents)
split_docs_2019=text_splitter.split_documents(computer_science_2019_documents)
split_docs_2016=text_splitter.split_documents(computer_science_2016_documents)
underlying_embeddings = OpenAIEmbeddings(openai_api_key='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')
store = LocalFileStore("./cache/")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace=underlying_embeddings.model
)
db_all=FAISS.from_documents(documents=split_docs_all,embedding=cached_embedder)
retriever_all=db_all.as_retriever(
    search_kwargs={"k": 6}
)
retriever_all_cosine=db_all.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.60}
)
syllabus_loader=PyPDFLoader('/Users/LakshSarda/PycharmProjects/pythonProject3/Computer_Science_Chatbot/CS_chatbot/595424-2023-2025-syllabus.pdf')
syllabus=syllabus_loader.load()
text_splitter_2=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
split_syllabus=text_splitter_2.split_documents(syllabus)
underlying_embeddings_2 = OpenAIEmbeddings(openai_api_key='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')
store = LocalFileStore("./cache/")
cached_embedder_2 = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings_2, store, namespace=underlying_embeddings_2.model
)
db_syllabus=FAISS.from_documents(documents=split_syllabus,embedding=cached_embedder_2)
retriever_syllabus_1 = db_syllabus.as_retriever(
        search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.60}
    )
def cs_ms(question):
    
    llm=ChatOpenAI(openai_api_key='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',temperature=0,model_name='gpt-4-turbo-preview')
    retrieved_docs=retriever_all_cosine.get_relevant_documents(
        question
    )
    print (retrieved_docs)

    relevant_sources = [
        {
            'page_content': doc.page_content,
            'metadata': doc.metadata['source']
        }
        for doc in retrieved_docs
    ]
    index=0
    pdf_of_ms=[]
    corresponding_page_content=[]
    for index, source in enumerate(relevant_sources):
        metadata_path_parts = source['metadata'].split('/')
        if "qp" in metadata_path_parts[-1]:
            # First type format
            metadata_path_parts[-2] = metadata_path_parts[-2].replace('question_papers','marking_scheme')
            metadata_path_parts[-1] = metadata_path_parts[-1].replace('qp', 'ms')
        else:
            # Second type format
            metadata_path_parts[-2] = 'marking_scheme'
            metadata_path_parts[-1] = metadata_path_parts[-1].replace('question_papers', 'marking_scheme')
            metadata_path_parts[-1] = metadata_path_parts[-1].replace('.pdf', '-mark-scheme.pdf')

        modified_metadata = '/'.join(metadata_path_parts)
        corresponding_page_content.append(source['page_content'])
        pdf_of_ms.append(modified_metadata)
        index=index+1
    loaded_documents = []
    for path in pdf_of_ms:
        loader_2=PyPDFLoader(path)
        documents=loader_2.load()
        loaded_documents.extend(documents)
    paper_codes=[]
    for document in loaded_documents:
        paper_source = document.metadata['source']
        paper_code = paper_source.split('/')[-1]
        paper_codes.append(paper_code)
    split_docs_2=text_splitter_2.split_documents(loaded_documents)
    db_new=FAISS.from_documents(documents=split_docs_2,embedding=cached_embedder)
    retriever_end=db_new.as_retriever(
    search_kwargs={"k": 15}
)
    syllabus_template="""Use the following context to answer the question:
    You are a component of a chatbot specifically designated to output all content in the IGCSE computer_science
    Syllabus corresponding to the user's question. This is the process you should follow:
    1. Identify key words in the user's question which could include technical computer_science terms etc.
    2. Search the syllabus and find content corresponding to the key word.
    3. Output that content as it is without adding your own information.
    4. If nothing relevant was found in the syllabus, say "nothing relevant was found in the syllabus." Please do not add your own
    points and keep responses as short as possible and precise to user question.
    {context}
    User_question: {question}
    Syllabus_content:"""
    prompt_syllabus=PromptTemplate.from_template(syllabus_template)
    ms_template="""Use the following context to answer the question:
    You are a chatbot specifically designed to give answers to user questions in exact answer-key language in Computer Science
    for the IGCSE Academic curriculum. Here are the steps you should follow to do the same:
    1. Based on the user question, find the content in the answer keys given to you which is most likely to be the answer to the user's
    question. Do not leave out points from the relevant answer you select. 
    2. Output all content which is the answer to user question from the answer keys. 
    3. Match the relevant content you selected in step 1 to your own knowledge of the question and only go forward with
    outputting the relevant content if the content of the answer key is somewhat similar to your own understanding of what the answer should be.
    4.  If relevant documents were retrieved,  you must try to output an answer from the answer key as it is without missing out points.
        Please only answer according to the answer keys provided without changing its wording and adding your own information and points. 
        Please make sure that the answer you output is factually correct by making sure that the answer makes sense to you- a computer
         science teacher.
    Output all content even if the answer key says "any two from" or something similar. Only limit your number of points if the user asks
    so. DO NOT MAKE UP YOUR OWN CONTENT, ANSWER FROM ONLY THE RELEVANT ANSWER KEY.

    
    {context}
    user_question: {question}
    Helpful answer based on guidelines above: 
    """
    prompt_ms=PromptTemplate.from_template(template=ms_template)
    output_parser=StrOutputParser()
    chain=ConversationalRetrievalChain.from_llm(
        llm=llm_google,
        retriever=retriever_end,
        combine_docs_chain_kwargs={"prompt": prompt_ms},
        memory=ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=0),
    )
    chain_syllabus=ConversationalRetrievalChain.from_llm(
        llm=llm_google,
        combine_docs_chain_kwargs={"prompt":prompt_syllabus},
        memory=ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=0),
        retriever=retriever_syllabus_1,
        
    )
    print (retriever_end.invoke(question))
    output=chain.invoke(question)
    print (output)
    output_s=chain_syllabus.invoke(question)
    answeronly_s=output_s.get('answer','')
    answeronly=output.get('answer','') 
    final_ms=output_parser.invoke(answeronly)
    final_syllabus=output_parser.invoke(answeronly_s)
    unique_paper_codes = list(set(paper_codes))

    # Convert unique paper codes list to a string
    paper_codes_string = ", ".join(unique_paper_codes)
    outputs="As per marking scheme: <br/><br/>"+final_ms+ "<br/><br/>Relevant syllabus content: <br/><br/>"+final_syllabus+"<br/><br/>Relevant answer keys: <br/><br/>"+paper_codes_string
    return outputs
app = Flask(__name__)
CORS(app)  
@app.route('/item', methods=['POST'])
def add_item():
    data = request.json  
    return cs_ms (data["question"]),201
if __name__ == '__main__':
    app.run(port=8002,debug=True)
