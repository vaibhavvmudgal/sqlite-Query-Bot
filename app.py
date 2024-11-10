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
from langchain_groq import ChatGroq

st.set_page_config(page_title="Chat with QueryEase", page_icon="🦜")
st.title("🦜 Chat with QueryEase")

# Default API Key
API_KEY = "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"

# Database Options
radio_opt = ["Use SQLite 3 Database", "Connect to your MySQL Database"]
selected_opt = st.sidebar.radio(label="Choose the DB which you want to chat", options=radio_opt)

# Database URI setup
if radio_opt.index(selected_opt) == 1:
    db_uri = "USE_MYSQL"
    mysql_host = st.sidebar.text_input("Provide MySQL Host")
    mysql_user = st.sidebar.text_input("MySQL User")
    mysql_password = st.sidebar.text_input("MySQL password", type="password")
    mysql_db = st.sidebar.text_input("MySQL database")
else:
    db_uri = "USE_LOCALDB"

# No need to ask for API key now as it's already set by default
api_key = API_KEY

# LLM model setup
llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)

# Function to handle file conversion
def create_sqlite_with_inferred_types(file, db_file):
    """Convert uploaded CSV or Excel file to SQLite database."""
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None

    # Create SQLite database from the DataFrame
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    df.to_sql('data', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()

# Function to configure the database
@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    if db_uri == "USE_LOCALDB":
        dbfilepath = (Path(__file__).parent / "student.db").absolute()
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    elif db_uri == "USE_MYSQL":
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MySQL connection details.")
            st.stop()
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))   

# Database connection
if db_uri == "USE_MYSQL":
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
else:
    db = configure_db(db_uri)

# Toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Agent with handle_parsing_errors set to True and only final answers returned
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    return_final_only=True  # Ensures agent stops after final answer is given
)

# Chat setup
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Enter another query")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[streamlit_callback])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)

# File Upload: Allow user to upload .db, .csv, or .xlsx
uploaded_file = st.sidebar.file_uploader("Upload your Database (SQLite, CSV, Excel)", type=["db", "sqlite", "csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(('csv', 'xlsx')):
        st.info("Your file will be converted into a SQLite database.")
        dbfilepath = "converted_db.db"
        create_sqlite_with_inferred_types(uploaded_file, dbfilepath)
        # Provide the SQLite file for download
        with open(dbfilepath, "rb") as f:
            st.download_button("Download SQLite File", f, file_name=dbfilepath, mime="application/octet-stream")
    elif uploaded_file.name.endswith(('db', 'sqlite')):
        dbfilepath = Path(uploaded_file.name)
        with open(dbfilepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        # Use the uploaded file for database interaction
        db = configure_db("USE_LOCALDB")
    else:
        st.error("Unsupported file type uploaded. Please upload a .db, .sqlite, .csv, or .xlsx file.")
