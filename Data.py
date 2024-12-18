import pandas as pd
from impala.dbapi import connect

# Function to search for columns in a specific Impala database
def search_columns(impala_host, impala_port, database_name):
    # Connect to Impala
    conn = connect(host=impala_host, port=impala_port)
    cursor = conn.cursor()

    # List to store results
    results = []

    # Get all tables in the specified database
    cursor.execute(f"SHOW TABLES IN {database_name}")
    tables = cursor.fetchall()

    # Loop through each table and get columns and their data types
    for table in tables:
        table_name = table[0]
        cursor.execute(f"DESCRIBE {database_name}.{table_name}")
        columns = cursor.fetchall()

        # Collect column names and data types
        for column in columns:
            column_name = column[0]
            column_type = column[1]
            results.append((database_name, table_name, column_name, column_type))

    # Close the connection
    cursor.close()
    conn.close()

    return results

# Function to create an Excel file with the results
def create_excel_file(impala_host, impala_port, database_name, output_file='columns_info.xlsx'):
    columns_found = search_columns(impala_host, impala_port, database_name)

    # Create a DataFrame to save to Excel
    df = pd.DataFrame(columns_found, columns=['Database Name', 'Table Name', 'Column Name', 'Data Type'])

    # Save to Excel file
    df.to_excel(output_file, index=False)
    print(f"Excel file '{output_file}' created successfully.")

# Example Usage
if __name__ == "__main__":
    impala_host = 'your_impala_host'  # Replace with your Impala host
    impala_port = 21050  # Default Impala port
    database_name = 'your_database_name'  # Replace with the desired database name

    # Generate and save the Excel file
    create_excel_file(impala_host, impala_port, database_name)
