import csv

def convert_txt_to_csv(input_file_path, output_file_path):
    """
    Converts a text file to a CSV file with a 'Sentence' column header.
    Skips empty lines in the input file.
    """
    with open(input_file_path, 'r') as text_file, open(output_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Sentence'])  # Write the header row

        for line in text_file:
            stripped_line = line.strip()
            if stripped_line:  # Skip empty lines
                csv_writer.writerow([stripped_line])  # Write each non-empty line as a row in the CSV file

    print(f'Converted {input_file_path} to {output_file_path}')

# Paths to input and output files
input_file_path = 'input.txt'
output_file_path = 'spamtext.csv'

# Convert the text file to a CSV file
convert_txt_to_csv(input_file_path, output_file_path)
