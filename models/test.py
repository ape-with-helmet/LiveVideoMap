import h5py
try:
    with h5py.File('facenet_keras.h5', 'r') as file:
        # File is open and ready for reading
        print("File is valid and open.")
except IOError:
    # File could not be opened or is corrupted
    print("File is corrupted or not found.")