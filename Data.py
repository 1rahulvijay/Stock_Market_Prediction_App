from impala.dbapi import connect

# Function to search for columns in the Impala database
def search_columns(impala_host, impala_port, search_string):
    # Connect to Impala
    conn = connect(host=impala_host, port=impala_port)
    cursor = conn.cursor()

    # Query to get all databases in Impala
    cursor.execute("SHOW DATABASES")
    databases = cursor.fetchall()

    results = []

    # Loop through each database and search for columns
    for database in databases:
        db_name = database[0]
        # Get all tables in the current database
        cursor.execute(f"SHOW TABLES IN {db_name}")
        tables = cursor.fetchall()

        for table in tables:
            table_name = table[0]
            cursor.execute(f"DESCRIBE {db_name}.{table_name}")
            columns = cursor.fetchall()

            # Check if any column contains the search string
            for column in columns:
                column_name = column[0]
                if search_string.lower() in column_name.lower():
                    results.append((db_name, table_name, column_name))

    # Close the connection
    cursor.close()
    conn.close()

    return results

# Example Usage
if __name__ == "__main__":
    impala_host = 'your_impala_host'  # Replace with your Impala host
    impala_port = 21050  # Default Impala port
    search_string = 'your_search_string'  # Replace with the string you want to search for in column names

    columns_found = search_columns(impala_host, impala_port, search_string)

    # Print the results
    if columns_found:
        for db_name, table_name, column_name in columns_found:
            print(f"Database: {db_name}, Table: {table_name}, Column: {column_name}")
    else:
        print("No columns found matching the search string.")
