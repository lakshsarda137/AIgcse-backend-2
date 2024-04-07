from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from flask import Flask,jsonify,Request,request
from flask_cors import CORS
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain.embeddings import CacheBackedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.storage import LocalFileStore
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
output_parser=StrOutputParser()
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
llm_google = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key='AIzaSyCW71zfRU69nAknLobdkOqjM1noPqqlxG0',
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,

    },
    temperature=0.75
)
output_parser=StrOutputParser()

loader=DirectoryLoader('/Users/LakshSarda/PycharmProjects/pythonProject3/Computer_Science_Chatbot/Economics_chatbot/Explainer_econ',loader_cls=PyPDFLoader)
docs=loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
split_docs = text_splitter.split_documents(docs)
underlying_embeddings = OpenAIEmbeddings(openai_api_key='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')
store = LocalFileStore("./cache/")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace=underlying_embeddings.model
)
db=FAISS.from_documents(documents=split_docs,embedding=cached_embedder)
def economics_explainer(question,history):
    sentences=history
    start_index = max(len(sentences) - 6, 0)
    filtered_sentences = sentences[start_index:]
    result = ""
    for sentence in filtered_sentences:
        result += f"{sentence['content']}. This message was sent by the {sentence['role']}. "
    fin="Answer the question based on the following chat history: "+result+". The question is now: "+question 
    retriever_sim = db.as_retriever(
    search_kwargs={"k": 15}
)
    template="""Use the following context to answer user question:
    You are a component of a chatbot that helps explain concepts to students in Economics specific to only the IGCSE 
    academic curriculum like an IGCSE Economics teacher. You have been given a set of pdfs that explain various concepts relevant to the syllabus of Economics in IGCSE
    and you must use them to help the user understand their question's concept. Your explanation must explain any key terminology it uses. It
    is not necessary that everything related to Economics is in the notes. The notes are just confined to the iGCSE
    Economics syllabus and therefore your answers must also be confined to the IGCSE Economics syllabus. Please answer
    only according to the pdf. Make sure to look at the entire pdf points before giving your answer, you should not miss out on
    points and your answers should be comprehensive and long enough to help the user to understand the answer to their question properly.
    {context}
    Question: {question}
    history={chat_history}
    Helpful answer: """
    prompt=PromptTemplate.from_template(template=template)
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm_google,
        retriever=retriever_sim,
        combine_docs_chain_kwargs={"prompt": prompt},
        memory=ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=0)
    )
    # return (qa.invoke(question))
    output=qa.invoke(fin)
    print (retriever_sim.invoke(fin))
    answeronly=output.get('answer','')
    outputs=(output_parser.invoke(answeronly))
    return outputs  
app = Flask(__name__)
CORS(app)
@app.route('/item', methods=['POST'])
def add_item():
    data = request.json
    return economics_explainer (data["question"], data["memory"]),201
if __name__ == '__main__':
    app.run(port=8003,debug=True)


#Request format: 