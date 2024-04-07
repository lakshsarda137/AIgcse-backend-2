
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from flask import Flask,jsonify,Request,request
from flask_cors import CORS
from langchain_google_vertexai import ChatVertexAI
vertex=ChatVertexAI(project='chatbot-3793c',model_name='gemini-1.0-pro',temperature=0.0,max_output_tokens=2000)
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
    temperature=0.60,
)
output_parser=StrOutputParser()
economics_docs=DirectoryLoader('/Users/LakshSarda/PycharmProjects/pythonProject3/Computer_Science_Chatbot/Economics_chatbot/question_papers_econ_all',loader_cls=PyPDFLoader)
economics_documents=economics_docs.load()
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40,
    add_start_index=True
)

split_docs=text_splitter.split_documents(economics_documents)
underlying_embeddings = OpenAIEmbeddings(openai_api_key='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')
store = LocalFileStore("./cache/")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace=underlying_embeddings.model
)
syllabus_loader=PyPDFLoader('/Users/LakshSarda/PycharmProjects/pythonProject3/Computer_Science_Chatbot/Economics_chatbot/596945-2023-2025-syllabus.pdf')
syllabus=syllabus_loader.load()
text_splitter_2=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
db_all=FAISS.from_documents(documents=split_docs,embedding=cached_embedder)
retriever_all_cosine=db_all.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.35}
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
retriever_sim_2 = db_all.as_retriever(
    search_kwargs={"k": 7}
)
def econ_ms_4(question,predefined_history):
    retrieved_docs=retriever_sim_2.get_relevant_documents(
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
    You are a component of a chatbot specifically designated to output all content in the IGCSE economics
    Syllabus corresponding to the user's question. This is the process you should follow:
    1. Identify key words in the user's question which could include technical economics terms etc.
    2. Search the syllabus and find content corresponding to the key word.
    3. Output that content as it is without adding your own information.
    4. If nothing relevant was found in the syllabus, say "nothing relevant was found in the syllabus." Please do not add your own
    points and keep responses as short as possible and precise to user question.
    {context}
    User_question: {question}
    Syllabus_content:"""
    prompt_syllabus=PromptTemplate.from_template(syllabus_template)
    ms_template="""Use the following context to answer the user's question:
    You are a chatbot designed to give answers to the user's question based on the answer keys given to you of economics in the 
    IGCSE academic curriculum. Use these steps to do so:
    1. Based on the user's question, try to find the answer key content in the answer keys given to you by searching all the content
    in the documents retrieved for you and find the question for
    which the answer key content is directly relevant to the user's question.
    2. Based on the content you found in step-1, generate you own "framed answer" which encapsulates all points from the 
    relevant answer key points and conducts analysis of each point using economic terminology. In your framed answers, develop
    all points thoroughly and don't leave any of your points strangling and link them back to the question.
    . Do not leave out any points but do not add your own points either. Your job is to give
    answer key specific answers for the IGCSE economics curriculum, not general economics answers.
    -If the question contains identify, state or define, your answers should not be too long.
      -If the question contains 'discuss', you should evaluate both sides of the argument in the user's question where one
       paragraph supports "Why it might" and another paragraph tells all points of "Why it might not". These answers should be relatively long and should be able to obtain 8 marks.
      -If the command term is 'analyse', you need to develop your points from the answer key content with the help of the answer 
      key. These answers should gain a minimum of 6 marks.
    3. After giving your 'framed answer', give the 'Exact answer key content' that was relevant to the user's question as it is
    without adding your own points but without missing any points from the answer key. This answer key content must be 
     pasted as it is from the answer key. Please be sure that in the 'exact answer key content' section, all points are 
     pasted from the answer key, from the start of the question's answer key content to the end. Points cannot be missed.
    4. In summary, your response will contain two sections: one where you frame the answer based on the answer key's points and 
    one where you paste the relevant content as it is for the user's reference. 
       
      *DO NOT MISS OUT POINTS FROM THE RELEVANT ANSWER KEY WHILE GIVING YOUR FRAMED ANSWER AND GIVE YOUR ANSWER ACCORDING TO THE COMMAND TERM.
    
      
    {context}
     user_question: {question}
    Your_answer:
    
     
    """
    prompt_ms=PromptTemplate.from_template(template=ms_template)
    output_parser=StrOutputParser()
    chain=ConversationalRetrievalChain.from_llm(
        llm=vertex,
        retriever=retriever_sim,
        combine_docs_chain_kwargs={"prompt": prompt_ms},
        memory=ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=0),
    )
    chain_syllabus=ConversationalRetrievalChain.from_llm(
        llm=vertex,
        combine_docs_chain_kwargs={"prompt":prompt_syllabus},
        memory=ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=0),
        retriever=retriever_syllabus_1,     
    )
    output=chain.invoke(question)
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
   
    return econ_ms_4 (data["question"], data["memory"]),201


if __name__ == '__main__':
    app.run(port=8004,debug=True)


#Describe what happens to the rate of reaction when temp increases
#Explain, using your knowledge of structure and bonding, why carbonyl fluoride has a low melting point.
#What are the conditions for an object to be in equilibrium