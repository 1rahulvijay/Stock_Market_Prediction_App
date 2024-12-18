import pandas as pd
from impala.dbapi import connect

# Function to search for columns in the Impala database
def search_columns(impala_host, impala_port):
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

            # Check and collect column names and data types
            for column in columns:
                column_name = column[0]
                column_type = column[1]
                results.append((db_name, table_name, column_name, column_type))

    # Close the connection
    cursor.close()
    conn.close()

    return results

# Function to create an Excel file with the results
def create_excel_file(impala_host, impala_port, output_file='columns_info.xlsx'):
    columns_found = search_columns(impala_host, impala_port)

    # Create a DataFrame to save to Excel
    df = pd.DataFrame(columns_found, columns=['Database Name', 'Table Name', 'Column Name', 'Data Type'])

    # Save to Excel file
    df.to_excel(output_file, index=False)
    print(f"Excel file '{output_file}' created successfully.")

# Example Usage
if __name__ == "__main__":
    impala_host = 'your_impala_host'  # Replace with your Impala host
    impala_port = 21050  # Default Impala port

    # Generate and save the Excel file
    create_excel_file(impala_host, impala_port)
