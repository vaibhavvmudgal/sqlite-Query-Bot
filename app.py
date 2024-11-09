import sqlite3
import pandas as pd
import streamlit as st
from pathlib import Path

# Function to infer the SQLite column data types based on the content of the DataFrame
def infer_column_types(df):
    column_types = {}
    
    for column in df.columns:
        # Check the type of each column
        if df[column].dtype == 'object':
            # If the column is of type object, infer if it's numeric or text
            if df[column].apply(pd.to_numeric, errors='coerce').notnull().all():
                # If all values can be converted to numbers, classify it as INTEGER
                if df[column].apply(pd.to_numeric, errors='coerce').dropna().apply(float.is_integer).all():
                    column_types[column] = 'INTEGER'
                else:
                    column_types[column] = 'TEXT'  # Treat numbers as TEXT if there's mixed content
            else:
                column_types[column] = 'TEXT'
        elif df[column].dtype in ['int64', 'float64']:
            # For numeric columns, assign INTEGER
            column_types[column] = 'INTEGER'
        else:
            column_types[column] = 'TEXT'  # Default to TEXT for other types

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
            create_table_query += f"{column} VARCHAR, "
        else:
            create_table_query += f"{column} {column_type}, "
    
    # Remove the trailing comma and space
    create_table_query = create_table_query.rstrip(', ') + ')'
    
    # Execute the table creation
    cursor.execute(create_table_query)
    
    # Insert data into the table
    for _, row in df.iterrows():
        cursor.execute(f"INSERT INTO data ({', '.join(df.columns)}) VALUES ({', '.join(['?' for _ in df.columns])})", tuple(row))
    
    # Commit the changes and close the connection
    conn.commit()
    conn.close()

# Function to query the SQLite database
def query_database(db_file, query):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    try:
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        conn.close()
        return columns, results
    except sqlite3.Error as e:
        conn.close()
        return None, f"Error executing query: {e}"

# Streamlit App
def main():
    st.title("SQLite Query Bot")
    
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        dbfilepath = Path(uploaded_file.name).stem + ".sqlite"
        
        # Convert CSV/Excel to SQLite with inferred column types
        create_sqlite_with_inferred_types(uploaded_file, dbfilepath)
        st.success(f"File uploaded and converted to SQLite database: {dbfilepath}")
        
        # Display tables
        conn = sqlite3.connect(dbfilepath)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        conn.close()
        
        st.write("Available tables in the database:")
        st.write([table[0] for table in tables])
        
        # Query input
        query = st.text_area("Enter your SQL query", "SELECT * FROM data LIMIT 10;")
        
        if st.button("Execute Query"):
            columns, results = query_database(dbfilepath, query)
            if columns is not None:
                st.write(f"Query Results ({len(results)} rows):")
                st.write(columns)
                st.write(results)
            else:
                st.error(results)  # Display error message if query failed
    else:
        st.warning("Please upload a CSV or Excel file.")

if __name__ == "__main__":
    main()
