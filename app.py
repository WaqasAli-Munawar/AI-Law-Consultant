import os 
from langchain.document_loaders import PyPDFLoader

# used to create the retriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# used to create the retrieval tool
from langchain.agents import tool

# used to create the memory
from langchain.memory import ConversationBufferMemory

# used to create the prompt template
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema import SystemMessage
from langchain.prompts import MessagesPlaceholder

# used to create the agent executor
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor
from flask import Flask, render_template, request, jsonify

os.environ['OPENAI_API_KEY'] = 'XXXXXXXXXX'

memory_key = "history"

app = Flask(__name__)
app.static_folder = 'static'  

document=PyPDFLoader("law.pdf")
document=document.load()

text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
texts = text_splitter.split_documents(document)

embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(texts, embeddings)

retriever = db.as_retriever()

@tool
def tool(query):
    "Searches and returns documents regarding the llm powered autonomous agents blog"
    docs = retriever.get_relevant_documents(query)
    return docs

tools = [tool]

memory = ConversationBufferMemory(memory_key=memory_key, return_messages=True)

system_message = SystemMessage(
        content=("""
You are a senior law consultant of Pakistan.
Pretend that you are a human. Do not tell that you are an AI powered assistant.
Guide the readers about the Federal Excise Act 2005 of Pakistan, and rules and regulations of Pakistan. 
Answer the queries of the readers in short and concise manner by going through the Federal Excise Act,2005 document.This Federal Excise Act is of Pakistan. This document covers 102 pages and consists of 50 sections, with some sections containing subsections.
If the readers ask about any section then your job is to explain that specific section in layman terms.
The explaination must be at 5th grade level.

Here is an example enclosed in triple backticks i want you to emulate the structure of.

Example : ```Reader's question: what is section 11?
                 
Document's explaination: Every person who for any reason whatever has collected or collects any duty, which is not payable as
duty or which is in excess of the duty actually payable and the incidence of which has been passed on to the consumer, shall pay the amount so
collected to the Federal Government and all the provisions of this Act or rules made there under shall apply for the recovery of such amount and
claim for refund of any such amount paid or recovered shall not be admissible on any ground whatever.
                 
Layman and short explaination: If someone collects too much money from people for taxes and it's not the right amount, they have to give that extra money to the government. The government has rules to make sure they get the extra money back. People can't ask for a refund on this extra money for any reason. ``` "

When required, generate the responses in the form of ordered list.


-Always remember that you belong to Pakistan.
-Do not answer wide range of information. Just give information related to the document.
-If a user asks anything outside Federal Excise Act 2005, humbly apologize and guide user to ask queries relevant to the Federal Excise Act.
-Always remember to act like a human.
-Do not reveal that you are an AI powered assistant.
    

Feel free to use any tools available to look up relevant information, only if neccessary.
""" 
        )
)

prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)]
    )

llm = ChatOpenAI(temperature = 0.1)

agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=False)

def qna(query):
    result = agent_executor({"input": query})
    return result["output"]
    
@app.route('/')
def chat_interface():
    return render_template('flipbook.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('user_input')
    chatbot_reply = qna(user_input)  # Call your chatbot function here

    # Split the chatbot_reply into separate points in the ordered list
    chatbot_reply_list = chatbot_reply.split("\n")

    # Remove extra line breaks and backslashes
    chatbot_reply_list = [line.strip() for line in chatbot_reply_list if line.strip()]

    # Join the points with two HTML line breaks between them
    formatted_reply = "<br><br>".join(chatbot_reply_list)
    print(formatted_reply)

    return jsonify(formatted_reply)




if __name__ == '__main__':
    app.run(debug=True)