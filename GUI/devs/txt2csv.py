import csv

# Path to your input text file
input_file_path = 'input.txt'

# Path to your output CSV file
output_file_path = 'spamtext.csv'

# Open the input text file and the output CSV file
with open(input_file_path, 'r') as text_file, open(output_file_path, 'w', newline='') as csv_file:
    # Create a CSV writer object
    csv_writer = csv.writer(csv_file)

    # Iterate over each line in the text file
    for line in text_file:
        # Write the line as a row in the CSV file
        # Strip the newline character from the line and split it if needed
        csv_writer.writerow([line.strip()])

print(f'Converted {input_file_path} to {output_file_path}')
