import sqlite3
import pandas as pd
import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_groq import ChatGroq
import re

# Set Streamlit page config
st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="✿")
st.title("Chat with SQL DB")

# Upload database file or connect to MySQL
uploaded_file = st.sidebar.file_uploader("Upload your Database (SQLite, CSV, Excel)", type=["db", "sqlite", "csv", "xlsx"])
db_type = st.sidebar.selectbox("Database Type", ["SQLite", "MySQL"])

# Set the Groq API key directly in the code
api_key = "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"

# Function to infer column data types based on values in the column
def infer_column_types(df):
    column_types = {}
    for column in df.columns:
        if pd.api.types.is_integer_dtype(df[column]):
            column_types[column] = "INTEGER"
        elif pd.api.types.is_float_dtype(df[column]):
            column_types[column] = "REAL"
        elif pd.api.types.is_string_dtype(df[column]):
            column_types[column] = "TEXT"
        else:
            column_types[column] = "TEXT"
    return column_types

# Function to convert CSV or Excel to SQLite with inferred column types
def create_sqlite_with_inferred_types(file, db_file):
    # Read the uploaded file (either CSV or Excel)
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None
    
    # Infer column types
    column_types = infer_column_types(df)
    
    # Create a connection to the SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Create a table with inferred column types
    create_table_query = "CREATE TABLE IF NOT EXISTS data ("
    for column in df.columns:
        # Use the inferred types
        column_type = column_types.get(column, 'TEXT')  # Default to TEXT if no type inferred
        # For TEXT columns, use VARCHAR (this is a synonym in SQLite)
        if column_type == 'TEXT':
            create_table_query += f"`{column}` VARCHAR, "
        else:
            create_table_query += f"`{column}` {column_type}, "
    
    # Remove the trailing comma and space
    create_table_query = create_table_query.rstrip(', ') + ')'
    
    # Execute the table creation
    try:
        cursor.execute(create_table_query)
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Error while creating table: {e}")
        conn.close()
        return None
    
    # Insert data into the table
    for _, row in df.iterrows():
        insert_query = f"INSERT INTO data ({', '.join([f'`{col}`' for col in df.columns])}) VALUES ({', '.join(['?' for _ in df.columns])})"
        try:
            cursor.execute(insert_query, tuple(row))
        except sqlite3.Error as e:
            st.error(f"Error while inserting data: {e}")
            conn.close()
            return None
    
    # Commit the changes and close the connection
    conn.commit()
    conn.close()

# Initialize database object
db = None

if db_type == "SQLite" and uploaded_file is not None:
    dbfilepath = None
    
    if uploaded_file.name.endswith(("db", "sqlite")):
        dbfilepath = Path(uploaded_file.name)
        with open(dbfilepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        db = SQLDatabase.from_uri(f"sqlite:///{dbfilepath}")
    
    elif uploaded_file.name.endswith("csv") or uploaded_file.name.endswith("xlsx"):
        # Create SQLite database from CSV or Excel
        dbfilepath = "temp_db.db"  # Temporary SQLite file name
        create_sqlite_with_inferred_types(uploaded_file, dbfilepath)
        db = SQLDatabase.from_uri(f"sqlite:///{dbfilepath}")

elif db_type == "MySQL":
    engine = connect_mysql()
    if engine:
        db = SQLDatabase(engine)

def list_sqlite_tables(db_file_path):
    try:
        conn = sqlite3.connect(db_file_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        conn.close()
        return [table[0] for table in tables]
    except Exception as e:
        st.error(f"Error fetching table list: {e}")
        return []

def query_sqlite(db_file_path, query):
    try:
        conn = sqlite3.connect(db_file_path)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        st.error(f"Error executing query: {e}")
        return []

# Handle input query
def extract_name_from_query(query):
    """Extract name from the query, assuming it's part of 'Name = <value>' format."""
    # Use regex to find 'Name = ' followed by a string
    match = re.search(r"Name\s*=\s*'([^']+)'", query, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

if db:
    # Directly use the dbfilepath instead of db.uri
    db_file_path = dbfilepath if dbfilepath else "temp_db.db"

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

    # List tables
    tables = list_sqlite_tables(db_file_path)
    st.write("Tables in the database: ", tables)

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
                # Extract the name from the user input
                name = extract_name_from_query(user_query)
                
                if name:
                    # Querying the database manually with extracted name
                    query_result = query_sqlite(db_file_path, f"SELECT * FROM data WHERE Name = '{name}' LIMIT 10")
                    
                    # Check if the result is empty
                    if query_result:
                        response = pd.DataFrame(query_result, columns=["Column1", "Column2", "Column3"]).to_string(index=False)
                    else:
                        response = f"No records found for '{name}' in the 'data' table."

                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.write(response)
                else:
                    st.session_state.messages.append({"role": "assistant", "content": "Could not extract a valid name from your query."})
                    st.write("Could not extract a valid name from your query.")
            except Exception as e:
                st.error(f"Error during query execution: {e}")

else:
    st.info("Please upload a database file or connect to a MySQL database to start querying.")
    
