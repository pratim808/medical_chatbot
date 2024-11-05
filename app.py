from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings

from langchain_community.vectorstores import FAISS

from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)


embeddings = download_hugging_face_embeddings()

DB_FAISS_PATH = r'G:\Chatbot\data\vector'
print("vector_base is loading from the folder")
db = FAISS.load_local(DB_FAISS_PATH, embeddings,
                      allow_dangerous_deserialization=True)


# Create a ChatPromptTemplate
prompt = ChatPromptTemplate.from_template(prompt_template)

chain_type_kwargs = {"prompt": prompt}


# Load the GROQ keys
groq_api_key = os.getenv('GROQ_API_KEY')
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key,
               model_name="Llama3-8b-8192")


qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
