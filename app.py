import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
import pandas as pd
from io import BytesIO
from langchain_groq import ChatGroq

st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="✿")
st.title("Chat with SQL DB")

# Upload database file or connect to MySQL
uploaded_file = st.sidebar.file_uploader("Upload your Database (SQLite, CSV, Excel)", type=["db", "sqlite", "csv", "xlsx"])
db_type = st.sidebar.selectbox("Database Type", ["SQLite", "MySQL"])

# Set the Groq API key directly in the code
api_key = "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"

# Function to load CSV or Excel into a temporary SQLite database
def create_temp_sqlite_from_df(df, db_name="temp_db.db"):
    conn = sqlite3.connect(db_name)
    df.to_sql("data", conn, if_exists="replace", index=False)
    conn.close()
    return f"sqlite:///{db_name}"

# Configure and connect to MySQL
def connect_mysql():
    user = st.sidebar.text_input("MySQL Username")
    password = st.sidebar.text_input("MySQL Password", type="password")
    host = st.sidebar.text_input("MySQL Host", value="localhost")
    db_name = st.sidebar.text_input("MySQL Database Name")
    if st.sidebar.button("Connect to MySQL"):
        return create_engine(f"mysql+pymysql://{user}:{password}@{host}/{db_name}")
    return None

db = None  # Initialize db as None to handle cases where it may not get assigned

if db_type == "SQLite" and uploaded_file is not None:
    if uploaded_file.name.endswith(("db", "sqlite")):
        dbfilepath = Path(uploaded_file.name)
        with open(dbfilepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        db = SQLDatabase.from_uri(f"sqlite:///{dbfilepath}")

    elif uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)
        db_uri = create_temp_sqlite_from_df(df)
        db = SQLDatabase.from_uri(db_uri)

    elif uploaded_file.name.endswith("xlsx"):
        df = pd.read_excel(uploaded_file)
        db_uri = create_temp_sqlite_from_df(df)
        db = SQLDatabase.from_uri(db_uri)

elif db_type == "MySQL":
    engine = connect_mysql()
    if engine:
        db = SQLDatabase(engine)

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

    if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_query = st.chat_input(placeholder="Ask anything from the database")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            streamlit_callback = StreamlitCallbackHandler(st.container())
            response = agent.run(user_query, callbacks=[streamlit_callback])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
else:
    st.info("Please upload a database file or connect to a MySQL database to start querying.")
