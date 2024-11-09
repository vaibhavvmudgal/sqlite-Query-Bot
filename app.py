import streamlit as st
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_groq import ChatGroq
from io import BytesIO
import tempfile
import os
from pathlib import Path  

st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="✿")
st.title("Chat with SQL Database")

# Set your Groq API Key
api_key = "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"

# Sidebar for file upload and DB type selection
uploaded_file = st.sidebar.file_uploader("Upload your Database (SQLite, CSV, Excel)", type=["db", "sqlite", "csv", "xlsx"])
db_type = st.sidebar.selectbox("Database Type", ["SQLite", "MySQL"])

# Function to create a temporary SQLite database from CSV or Excel
def create_temp_sqlite_from_df(df):
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    conn = sqlite3.connect(temp_db.name)
    df.to_sql("data", conn, if_exists="replace", index=False)
    conn.close()
    return temp_db.name

# Function to connect to MySQL
def connect_mysql():
    user = st.sidebar.text_input("MySQL Username")
    password = st.sidebar.text_input("MySQL Password", type="password")
    host = st.sidebar.text_input("MySQL Host", value="localhost")
    db_name = st.sidebar.text_input("MySQL Database Name")
    if st.sidebar.button("Connect to MySQL"):
        return create_engine(f"mysql+pymysql://{user}:{password}@{host}/{db_name}")
    return None

db = None  # Initialize db as None to handle cases where it may not get assigned
temp_db_path = None  # Placeholder for SQLite file path

# Set up the database based on user input
if db_type == "SQLite" and uploaded_file is not None:
    if uploaded_file.name.endswith(("db", "sqlite")):
        # Handle uploaded SQLite database directly
        dbfilepath = Path(uploaded_file.name)
        with open(dbfilepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        db = SQLDatabase.from_uri(f"sqlite:///{dbfilepath}")

    elif uploaded_file.name.endswith("csv"):
        # Convert CSV to SQLite and load it
        df = pd.read_csv(uploaded_file)
        temp_db_path = create_temp_sqlite_from_df(df)
        db = SQLDatabase.from_uri(f"sqlite:///{temp_db_path}")

    elif uploaded_file.name.endswith("xlsx"):
        # Convert Excel to SQLite and load it
        df = pd.read_excel(uploaded_file)
        temp_db_path = create_temp_sqlite_from_df(df)
        db = SQLDatabase.from_uri(f"sqlite:///{temp_db_path}")

    # Offer download of the temporary SQLite database
    if temp_db_path:
        with open(temp_db_path, "rb") as file:
            btn = st.download_button(
                label="Download Converted SQLite Database",
                data=file,
                file_name="converted_database.db",
                mime="application/x-sqlite3"
            )

elif db_type == "MySQL":
    # Connect to MySQL database
    engine = connect_mysql()
    if engine:
        db = SQLDatabase(engine)

if db:
    # Initialize LLM with Groq's API and LangChain's toolkit
    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )

    # Display available tables in the database
    try:
        table_names = db.get_table_names()
        if not table_names:
            st.error("No tables found in the uploaded database.")
        else:
            st.write("**Available tables in the database:**")
            st.write(table_names)

            # Show schema for each table
            for table in table_names:
                try:
                    st.write(f"**Schema for `{table}` table:**")
                    schema = db.get_table_info(table)
                    st.write(schema)
                except Exception as e:
                    st.warning(f"Could not retrieve schema for table `{table}`. Error: {e}")
    except ValueError as e:
        st.error(f"Error loading tables: {e}")

    # Chat input and display
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
            try:
                response = agent.run(user_query, callbacks=[streamlit_callback])
            except ValueError as e:
                response = "An error occurred while processing your request. Please try again or rephrase your query."
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
else:
    st.info("Please upload a database file or connect to a MySQL database to start querying.")
