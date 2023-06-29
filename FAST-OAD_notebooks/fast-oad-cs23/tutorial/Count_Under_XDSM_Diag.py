def Count_Under_XDSM_Diag(file_path):
    """
    This counts the terms below the diagonal of a given aircraft sizing xsml.html file.
    The aim is to determine how slow the design loop will that be since it is almost proportional to the number of terms below the diagonal
    """

    import re
    #import os

    #file_path = os.path.abspath('FAST-OAD_notebooks/fast-oad-cs23/tutorial/xdsm.html') #change into generic path

    # Read the HTML file
    with open(file_path) as file:
        html_content = file.read()

    variables_pattern = r"'from':\s*'(\d+)',\s*'to':\s*'(\d+)',\s*'name':\s*'([^']+)'"
    edges_index = html_content.rfind('edges')  # Find the last occurrence of 'edges'

    # Extract the relevant portion of the script starting from 'edges'
    relevant_content = html_content[edges_index:]

    #Find all the instances of the pattern that indicates a variable in the relevant section of the xdsm.html
    variables = re.findall(variables_pattern, relevant_content)

    #Define empty matrix of the aproppriate size regardless of the particular index in the html file
    matrix_size = int(variables[-1][0]) - int(variables[0][0]) +1
    matrix = [[0] * matrix_size for _ in range(matrix_size)]

    # Iterate over the variables
    for variable in variables:
        from_index = int(variable[0])- int(variables[0][0])  # Extract the 'from' index as integer (this is the row number)
        to_index = int(variable[1]) - (int(variables[0][1])-1)  # Extract the 'to' index as integer (this is the column number)
        parameter_names = variable[2].split(',')  # Split the parameter names by comma
        # Update the matrix with the count of commas for each position
        matrix[from_index][to_index] += len(parameter_names) #Count how many design variables are in each of the posiitions of the xdsm matrix

    # Remove the first row and first column (these are not looping variables)
    matrix = [row[1:] for row in matrix[1:]]

    # Compute the sum of values below the diagonal (loops)
    total_var_in_loops = sum(matrix[i][j] for i in range(len(matrix)) for j in range(i))
    
    return total_var_in_loops

