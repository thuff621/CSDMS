import numpy as np

def rotate_asc_file(file_path, degrees):
    # Load the file into a numpy array
    data = np.loadtxt(file_path, skiprows=8)

    data = np.where(data.astype(str) == 'nan', 0, data)


    # Rotate the array by the specified number of degrees
    rotated_data = np.rot90(data, k=degrees // 90)

    # Save the rotated data back to the file
    with open(str(file_path).replace(".asc", '_rotated.asc'), 'w') as f:
        # Write the header
        with open(file_path, 'r') as f_header:
            header = f_header.read().splitlines()
            for i in range(6):
                f.write(header[i] + '\n')

        # Write the rotated data
        np.savetxt(f, rotated_data, fmt='%d')

# Example usage
rotate_asc_file(r'C:\Users\u4eewtph\Documents\workSpace\models\MarshMorpho2D\Thomas_pyMarshMorpho2D\pymarshmorpho2d-master\examples\Gull_Island_50meter.asc', -45)