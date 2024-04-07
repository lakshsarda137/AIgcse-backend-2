from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from flask import Flask,jsonify,Request,request
from flask_cors import CORS
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import ChatVertexAI
vertex=ChatVertexAI(project='chatbot-3793c',model_name='gemini-1.0-pro',temperature=0.0,max_output_tokens=2000)
from langchain.schema import StrOutputParser
from langchain.storage import LocalFileStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
output_parser=StrOutputParser()
llm=ChatOpenAI(openai_api_key='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',temperature=0.7,model_name='gpt-4-turbo-preview')
loader=DirectoryLoader('/Users/LakshSarda/PycharmProjects/pythonProject3/Computer_Science_Chatbot/Economics paper 1 ms',loader_cls=PyPDFLoader)
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
def economics_explainer(question,predefined_history):
    llm_google=ChatGoogleGenerativeAI(google_api_key='AIzaSyCW71zfRU69nAknLobdkOqjM1noPqqlxG0',temperature=0.75,model='gemini-1.0-pro')
    
    # The storage layer for the parent documents
    retriever_sim = db.as_retriever(
    search_kwargs={"k": 15}
)
    
#     template="""Use the following context to answer the user's question:
# You are a chatbot designed to frame answers in Economics for the IBDP.
# 1. Look at all documents given to you and find the question corresponding to which the answer key content contains points to answer
# the user's question.
# 2. Based on the points given in the answer key corresponding to the question, generate content in separate paragraphs to satisfy those
# points.
# 3. In case the point includes the construction of a diagram, then just specify what all to show in the diagram in your framed answer under
# "For your diagram, create a"
# 4. If the answer key content contains a point for evaluation, then evaluate whatever you suggest in your answer by telling the 
# impact on stakeholders, long run and short run impact, alternate policies.
# 5. After framing your answer, copy paste the answer key points you used as it is under "*Exact answer key content"*
# Make sure that you satisfy all points given in the relevant answer key. If the question possesses 10 marks, include a defintion of the key economic terms, a description
# of the diagram relevant for the question and an explanation of relevant economic theory using a made up example. If the question possessed 15 marks, your answer must contain an
# evaluation as a separate paragraph in addition to all requirements in the 10 marker question. Write your answer like a teacher.
# In case you are working on a 15 mark question (where comamand term is "Discuss" or "To what extent"), then your answer should be formed
# around your real world example. This means that there must be a lot of application of your real world example.
#     {context}
#     Question: {question}

#     Helpful answer: """
    template="""
    Use the following context to answer the user's question:
You are a chatbot designed to give framed answers to students studying IBDP Economics based on the relevant answer key. Follow these steps:
1. Search all retrieved documents and find which question’s answer key is relevant to the user’s question.
2. Create a “Framed Answer” section. 
3. Based on the first point given in the answer key corresponding to what answers should include, generate your first paragraph. This paragraph typically includes the definition to key economic terminology.
4.  Move on to the next point, and generate content according to that like a teacher would when writing an answer. If the question is worth 15 marks (determined by the command term, which is either “Discuss” or “To what extent”), then all your paragraphs and explanations other than the definition paragraph should be formed around a real world example of the question given to you. 
5. For a point stating to make a diagram, suggest what to include on the X-axis of the diagram, Y-axis, and what curves to make in accordance with what the answer key expects the diagram to contain. Your diagram recommendation must be based on the answer key’s expectations.
6. In case the answer key demands an evaluation, then use this format for your evaluation:
Outline the impact on various stakeholders.
Outline the short run and long run impact of whatever policy or statement is included in the question.
Outline alternate policies to the one given before.
7. You must make sure your answer satisfies and elaborates upon all points given in the relevant answer key using the helpful guidelines
I gave above. Separate your answer into distinct paragraphs where each new paragraph satisfies a given point in the answer key. Each
Paragraph must have the answer key point it is satisfying as its heading. 
Write your answer as an Economics teacher would. 
8. Below your "Framed Answer" section, give the "Exact answer key content" where you paste the relevant answer key content corresponding
to the question as it is from the answer key without making alterations.
You must ensure that your framed answer is formed around the Real world example you chose based on the question. Your diagram's axes
should also be labelled accordingly and your explanation should be generated accordingly. Make sure you only choose one real world
example for one question, and form your entire answer around that one real world example to give a good application of the real world
example in your essay.
Make sure that your framed answer does not just state points, but analyses them like an economist would using the common example used
throughout the answer.
These criterias must be satisfied for a 15 mark essay response:
1. Relevant economic terms are clearly defined.
2. Relevant economic theory is clearly explained and applied. 
3. Where appropriate, diagrams are included and applied effectively. 
4. Where appropriate, examples are used effectively.
5. There is evidence of appropriate synthesis or evaluation.
6. There are no significant errors.
NOTE: FOR YOUR REAL WORLD EXAMPLE IN A 15 MARKER, MENTION THE COUNTRY, YEAR AND EXACT FIGURES OF THE DATA YOU ARE USING FOR YOUR ANSWER.
    {context}
    Question: {question}

    Helpful answer:
"""
#     template="""
#     Use the following context to answer the user's question:
#     You are a chatbot designed to frame answers in Economics for the IBDP.
# 1. Look at all documents given to you and find the question corresponding to which the answer key content contains points to answer
# the user's question.
#     In crafting a 15-mark essay for IB Economics following the DEEEDS format, begin by defining key terms relevant to the question
# , ensuring clarity on foundational concepts. For example, when discussing the impact of minimum wage laws, define "minimum wage" as the
#  lowest legal salary that employers can pay workers. Then, explain the economic theories that underpin your argument, such as how minimum 
#  wage laws can affect employment levels. Evaluate these theories by weighing their advantages and disadvantages, considering both s
#  hort-term and long-term implications for workers and employers along with the impact on various stakeholders. Use a real-world example, like the increase in minimum wage in Seattle
#  , to illustrate your points, providing concrete evidence of theoretical outcomes. Incorporate a relevant diagram, such as the labor market
#    model showing the effect of a minimum wage above the equilibrium wage, to visually demonstrate the impact on unemployment levels. 
#    Discuss the short-run implications, like potential job losses in low-skilled sectors, and long-run effects, including increased living 
#    standards and reduced poverty. Finally, propose alternative solutions or viewpoints, such as government subsidies for low-income workers
#    , to present a well-rounded analysis. This comprehensive approach ensures a deep exploration of the topic, balancing theoretical insights
#  with practical examples and critical evaluation for a high-scoring essay.
#      {context}
#     Question: {question}

#     Helpful answer:"""
    prompt=PromptTemplate.from_template(template=template)
    from langchain.chains import ConversationalRetrievalChain
    qa = ConversationalRetrievalChain.from_llm(
        llm=vertex,
        retriever=retriever_sim,
        combine_docs_chain_kwargs={"prompt": prompt},
        memory=ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=3)
    )

    # return (qa.invoke(question))
    output=qa.invoke(question)
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
    app.run(port=1561,debug=True)