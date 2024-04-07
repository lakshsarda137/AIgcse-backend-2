
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from flask import Flask,jsonify,Request,request
from langchain_google_vertexai import ChatVertexAI
chat = ChatVertexAI(project='chatbot-3793c', anthropic_version='vertex-2023-10-16', temperature=0.0,max_output_tokens=2040)

from flask_cors import CORS
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain.embeddings import CacheBackedEmbeddings
vertex=ChatVertexAI(project='chatbot-3793c',model_name='gemini-1.0-pro',temperature=0.0,max_output_tokens=2000)
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.storage import LocalFileStore
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
llm_google = ChatGoogleGenerativeAI(
    model="gemini-1.0-pro",
    google_api_key='AIzaSyCW71zfRU69nAknLobdkOqjM1noPqqlxG0',
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,

    },
    temperature=0.45,
)
output_parser=StrOutputParser()
physics_docs=DirectoryLoader('/Users/LakshSarda/PycharmProjects/pythonProject3/Computer_Science_Chatbot/Physics_chatbot/question_papers_phy4_all',loader_cls=PyPDFLoader)
physics_documents=physics_docs.load()
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=200,
    add_start_index=True
)
split_docs=text_splitter.split_documents(physics_documents)
underlying_embeddings = OpenAIEmbeddings(openai_api_key='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')
store = LocalFileStore("./cache/")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace=underlying_embeddings.model
)
syllabus_loader=PyPDFLoader('/Users/LakshSarda/PycharmProjects/pythonProject3/Computer_Science_Chatbot/Physics_chatbot/Physics syllabus.pdf')
syllabus=syllabus_loader.load()
text_splitter_2=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    add_start_index=True
)
db_all=FAISS.from_documents(documents=split_docs,embedding=cached_embedder)
retriever_all_cosine=db_all.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.35}
)
retriever_sim_2 = db_all.as_retriever(
    search_kwargs={"k": 8}
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
def phy_ms_4(question,predefined_history):
    retrieved_docs=retriever_all_cosine.get_relevant_documents(
        question
    )
    print (retrieved_docs)
    relevant_sources = [
        {
            'page_content': doc.page_content,
            'metadata': doc.metadata['source']  # Extracting the source information
        }
        for doc in retrieved_docs
    ]
    # Display the relevant sources
    index=0
    pdf_of_ms=[]
    corresponding_page_content=[]
    for index, source in enumerate(relevant_sources):

        # Check if it's the third source

            # Modify the source metadata
        metadata_path_parts = source['metadata'].split('/')

        # Check for the presence of "qp" in the metadata
        if "qp" in metadata_path_parts[-1]:
            # First type format
            metadata_path_parts[-2] = metadata_path_parts[-2].replace('question_papers','marking_scheme')
            metadata_path_parts[-1] = metadata_path_parts[-1].replace('qp', 'ms')
        else:
            # Second type format
            metadata_path_parts[-2] = 'marking_scheme'
            metadata_path_parts[-1] = metadata_path_parts[-1].replace('question_papers', 'marking_scheme')
            

        modified_metadata = '/'.join(metadata_path_parts)
        corresponding_page_content.append(source['page_content'])

        # Print the modified metadata for the third source

        pdf_of_ms.append(modified_metadata)

        index=index+1
    loaded_documents = []
    for path in pdf_of_ms:
        loader_2=PyPDFLoader(path)
        documents=loader_2.load()
        loaded_documents.extend(documents)
    #print (loaded_documents)
    paper_codes=[]
    for document in loaded_documents:
        paper_source = document.metadata['source']
        paper_code = paper_source.split('/')[-1]
        paper_codes.append(paper_code)


    split_docs_2=text_splitter_2.split_documents(loaded_documents)

    db_2=FAISS.from_documents(documents=split_docs_2,embedding=cached_embedder_2)
    
    retriever_sim = db_2.as_retriever(
    search_kwargs={"k": 15}
)
    syllabus_template="""Use the following context to answer the question:
    You are a component of a chatbot specifically designated to output all content in the IGCSE physics
    Syllabus corresponding to the user's question. This is the process you should follow:
    1. Identify key words in the user's question which could include technical physics terms etc.
    2. Search the syllabus and find content corresponding to the key word.
    3. Output that content as it is without adding your own information.
    4. If nothing relevant was found in the syllabus, say "nothing relevant was found in the syllabus." Please do not add your own
    points and keep responses as short as possible and precise to user question.
    {context}
    User_question: {question}
    Syllabus_content:"""
    prompt_syllabus=PromptTemplate.from_template(syllabus_template)
    ms_template="""Use the following context to answer the question:
    You are a chatbot specifically designed to give answers to user questions in exact answer-key language in physics
    for the IGCSE Academic curriculum. Here are the steps you should follow to do the same:
    1. Based on the user question, find the content in the answer keys given to you which is most likely to be the answer to the user's
    question. Do not leave out points from the relevant answer you select.  
    2. Match the relevant content you selected in step 1 to your own knowledge of the question and only go forward with
    outputting the relevant content if the content of the answer key is somewhat similar to your own understanding of what the answer should be.
    4.  If relevant documents were retrieved,  you must try to output an answer from the answer key as it is without missing out points and adding your
    own information.
        Please only answer according to the answer keys provided without changing its wording and adding your own information and points. 
        Please make sure that the answer you output is factually correct by making sure that the answer makes sense to you- a physics
         eacher.
    Output all content even if the answer key says "any two from" or something similar. Only limit your number of points if the user asks
    so. Do not miss out on any points from the relevant answer key that you choose but do not add your own points.
    Note: Do not add your own points or information, just use the content given in the relevant answer key.
    
    {context}
    user_question: {question}
    Helpful answer based on guidelines above: 
    """
    prompt_ms=PromptTemplate.from_template(template=ms_template)
    output_parser=StrOutputParser()
    chain=ConversationalRetrievalChain.from_llm(
        llm=vertex,
        retriever=retriever_sim,
        combine_docs_chain_kwargs={"prompt": prompt_ms},
        memory=ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=0),
    )
    print (retriever_sim.invoke(question))
    chain_syllabus=ConversationalRetrievalChain.from_llm(
        llm=chat,
        combine_docs_chain_kwargs={"prompt":prompt_syllabus},
        memory=ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=0),
        retriever=retriever_syllabus_1,     
    )
    output=chain.invoke(question)
    print (output)
    output_s=chain_syllabus.invoke(question)
    answeronly_s=output_s.get('answer','')
    answeronly=output.get('answer','') 
    final_ms=output_parser.invoke(answeronly)
    print (final_ms)
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
    return phy_ms_4 (data["question"], data["memory"]),201
if __name__ == '__main__':
    app.run(port=8008,debug=True)
