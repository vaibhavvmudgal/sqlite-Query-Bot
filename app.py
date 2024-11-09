import streamlit as st
from pathlib import Path
import pandas as pd
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq

st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="✿")
st.title("Chat with SQL DB")

# Set the Groq API key directly in the code
api_key = "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"

# Option to upload file or provide MySQL URI
uploaded_file = st.sidebar.file_uploader("Upload your SQLite, CSV, or Excel file", type=["db", "sqlite", "csv", "xlsx"])
mysql_uri = st.sidebar.text_input("Or enter your MySQL Database URI (e.g., mysql+pymysql://user:password@host/dbname)")

# Function to convert uploaded file to an SQLite database
@st.cache_resource(ttl="2h")
def configure_db(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        dbfilepath = Path(uploaded_file.name)
        with open(dbfilepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))

    # Create a temporary SQLite database for CSV/Excel
    temp_db_path = ":memory:"
    conn = sqlite3.connect(temp_db_path)
    df.to_sql("uploaded_data", conn, index=False, if_exists="replace")
    conn.close()
    return SQLDatabase(create_engine("sqlite:///:memory:", creator=lambda: sqlite3.connect(temp_db_path)))

# Configure either SQLite or MySQL database based on input
if uploaded_file is not None:
    db = configure_db(uploaded_file)
elif mysql_uri:
    db = SQLDatabase(create_engine(mysql_uri))
else:
    st.info("Please upload a database file or enter a MySQL URI to start querying.")
    db = None

if db:
    # LLM model
    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)

    # Toolkit
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # Agent
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )

    # Initialize or clear chat history
    if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    # Display chat messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # User input for query
    user_query = st.chat_input(placeholder="Ask anything from the database")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            streamlit_callback = StreamlitCallbackHandler(st.container())
            response = agent.run(user_query, callbacks=[streamlit_callback])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
