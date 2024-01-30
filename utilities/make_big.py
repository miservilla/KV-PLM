def append_lines_to_file(file_path, n):
    # Step 1: Read the content of the file
    with open(file_path, 'r') as file:
        line = file.readline().strip()

    # Step 2 and 3: Append the line n times
    with open(file_path, 'a') as file:
        for _ in range(n):
            file.write(line + '\n')

# Usage example
file_path = 'Ret/test.txt'  # Replace with your file path
n = 15000 # Number of times to append the line
append_lines_to_file(file_path, n)
