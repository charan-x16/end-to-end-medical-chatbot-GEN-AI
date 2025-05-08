from flask import Flask, render_template, jsonify, request
from src.helper import download_huggingface_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os 


load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENROUTER_API_KEY = "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"


embeddings = download_huggingface_embeddings()
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(embedding=embeddings,
                                                    index_name=index_name)


retriever = docsearch.as_retriever(search_type='similarity', search_kwargs={'k':3})


llm = ChatOpenAI(model="mistralai/mistral-small-3.1-24b-instruct:free",  api_key=OPENROUTER_API_KEY,
                 base_url="https://openrouter.ai/api/v1",)




system_prompt = (
    "You are an assistant for question-answering tasks."
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you"
    "don't know. Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)


prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


app = Flask(__name__, template_folder='template')

@app.route('/')
def index():
    return render_template("chat.html")

@app.route('/get', methods=['GET', 'POST'])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({'input': msg})
    print('Response: ', response['answer'])
    return str(response['answer'])


if __name__ == '__main__':
    app.run(debug=True)