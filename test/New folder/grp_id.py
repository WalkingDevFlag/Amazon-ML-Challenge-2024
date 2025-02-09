import pandas as pd

# Load the CSV file
csv_file = r'E:\Random Python Scripts\Amazon ML Challenge\Dataset\train.csv'  # Replace with your actual train.csv path
df = pd.read_csv(csv_file)

# Group by 'group_id' and count the number of rows for each unique group_id
group_counts = df['group_id'].value_counts()

# Save the result to a text file
output_file = 'group_id_counts.txt'  # The path to the text file where you want to save the result
with open(output_file, 'w') as f:
    for group_id, count in group_counts.items():
        f.write(f"group_id: {group_id}, count: {count}\n")

print(f"Group ID counts have been saved to {output_file}")
