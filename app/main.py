import streamlit as st
from dotenv import dotenv_values
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from datetime import datetime
import chromadb
from langchain_openai import AzureOpenAIEmbeddings

config = dotenv_values(".env")
embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-small",
    azure_endpoint=config["AZURE_EMBEDDING_ENDPOINT"],
    api_key=config["AZURE_EMBEDDING_API_KEY"],
)

ef = embeddings.embed_documents
st.title("LangChain practical example")

groq_api_key = config["GROQ_API_KEY"]

client = chromadb.HttpClient(host="localhost", port=8765)
# client = chromadb.PersistentClient(path="chroma_db")
if not (ret := client.heartbeat()):
    st.error("Chroma server is not running")
    st.stop()
else:
    st.write(f"Chroma time: {datetime.fromtimestamp(int(ret / 1e9))}")


collection = client.get_or_create_collection("war_and_peace_local")


system_message = SystemMessage(
    content="""You are a helpful assistant.
Only answer questions based on the results from below. If you don't know the answer, say you don't know."""
)
if "latest_msgs_sent" not in st.session_state:
    st.session_state.latest_msgs_sent = []
if "file_path" not in st.session_state:
    st.session_state.file_path = None

if "llm" not in st.session_state:
    llm = ChatGroq(
        model=config["GROQ_MODEL"],
        api_key=groq_api_key,
        temperature=0.7,
        # max_tokens=40,
    )
    st.session_state.llm = llm
else:
    llm = st.session_state.llm


if "messages" not in st.session_state:
    st.session_state.messages = []


def get_relevant_docs(message: HumanMessage) -> list[str]:
    # Query database for relevant docs
    pass
    # Return list of relevant docs
    return [Document(page_content="Foo bar"), Document(page_content="Baz qux")]


def generate_response(msg: str):
    res = collection.query(query_texts=[msg], n_results=5)
    docs = [i[0] for i in res["documents"]]
    start_time = datetime.now()
    # relevant_docs = get_relevant_docs(st.session_state.messages[-1])
    combined_sys_message = f"""{system_message}
Use the following documents to answer the question:
{"\n".join(docs)}"""
    # combined_sys_message = system_message
    messages = [SystemMessage(content=combined_sys_message)] + st.session_state.messages
    response = llm.invoke(messages)
    st.session_state.messages.append(response)
    st.session_state.latest_msgs_sent = messages
    return response


for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

if msg := st.chat_input("Enter a message"):
    st.session_state.messages.append(HumanMessage(content=msg))
    response = generate_response(msg)
    st.rerun()

# st.divider()
# st.write(st.session_state.file_path)
# st.write(st.session_state.latest_msgs_sent)
