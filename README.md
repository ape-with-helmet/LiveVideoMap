
# Face Recognition using TensorFlow and Keras

This project demonstrates a simple face recognition system using TensorFlow and Keras. It captures an image from a webcam, detects faces using the MTCNN (Multi-Task Cascaded Convolutional Networks) face detector, and compares the detected faces with a pre-trained FaceNet model.

## Installation

To install the required dependencies, follow these steps:

1. **Install Python 3.7 or higher:** [Download Python](https://www.python.org/downloads/)
2. **Create a virtual environment (optional):**
   - Install `virtualenv` using pip: 
     ```bash
     pip install virtualenv
     ```
   - Create a new virtual environment: 
     ```bash
     virtualenv venv
     ```
   - Activate the virtual environment:
     - On **Windows**: 
       ```bash
       venv\Scripts\activate
       ```
     - On **macOS/Linux**: 
       ```bash
       source venv/bin/activate
       ```
3. **Clone this repository:**
   ```bash
   git clone https://github.com/ape-with-helmet/LiveVideoMap.git
   ```
4. **Navigate to the project directory:**
   ```bash
   cd LiveVideoMap
   ```
5. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the face recognition program, execute the following command:

```bash
python face_feed.py
```

The program will open a window displaying the live video feed from your webcam. Press **'s'** to capture an image of an Aadhaar card face, and press **'q'** to quit the program.

The program will compare the captured Aadhaar card face with the detected faces in the live video feed and display the match results.

## Notes

- Make sure you have a compatible NVIDIA GPU and the appropriate CUDA drivers installed for TensorFlow GPU support.
- The pre-trained FaceNet model used in this example is provided by the `keras-facenet` library.
- The program uses the MTCNN face detector from the `mtcnn` library.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
