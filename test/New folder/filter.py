import pandas as pd

# Load the CSV file
csv_file = r'E:\Random Python Scripts\Amazon ML Challenge\Dataset\train.csv'
df = pd.read_csv(csv_file)

# Filter rows where group_id is 731432
filtered_df = df[df['group_id'] == 731432]

# Get the number of cells where group_id is 731432
num_cells = filtered_df.size
print(f"Number of cells with group_id = 731432: {num_cells}")

# Save the filtered rows to a new CSV file
output_file = 'filtered_output.csv'  # Replace with desired output CSV file path
filtered_df.to_csv(output_file, index=False)

print(f"Filtered rows have been saved to {output_file}")
