# Chat-with-Pdf
# Chat with Multiple PDFs Application

## Overview
The "Chat with Multiple PDFs" application is a Streamlit-based tool that allows users to upload multiple PDF documents and interact with them through a conversational chatbot interface. Using this app, you can ask questions about the content of the PDFs, and the app will provide relevant answers by processing the text within the documents.

## Features
- **PDF Upload:** Upload multiple PDF documents for processing.
- **Text Extraction:** Extracts text content from the uploaded PDFs.
- **Chunking:** Splits the extracted text into manageable chunks for efficient processing.
- **Vector Search:** Utilizes FAISS (Facebook AI Similarity Search) for efficient text retrieval.
- **Conversational Interface:** Interact with the uploaded documents using a chatbot.
- **Customizable Templates:** Bot and user messages are styled with custom HTML and CSS.

## Requirements
### Libraries and Packages
Make sure you have the following Python packages installed:
- `streamlit`
- `PyPDF2`
- `langchain`
- `langchain-community`
- `sentence-transformers`
- `faiss-cpu`
- `python-dotenv`

You can install them using:
```bash
pip install streamlit PyPDF2 langchain langchain-community sentence-transformers faiss-cpu python-dotenv
```

### Environment Variables
To use Hugging Face Hub's API, you must set up an environment variable for the API key in a `.env` file:
```plaintext
HUGGINGFACEHUB_API_TOKEN=<your_huggingface_api_key>
```

## How to Use
1. Clone this repository and navigate to the project directory.
2. Install the required dependencies using the command:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory and add your Hugging Face API key.
4. Run the application with:
   ```bash
   streamlit run app.py
   ```
5. Upload your PDF documents using the sidebar uploader.
6. Click the "Process" button to process the uploaded PDFs.
7. Ask questions about the documents in the input box, and the chatbot will provide answers.

## File Structure
```
|-- app.py             # Main application script
|-- .env               # Environment file for Hugging Face API key
|-- requirements.txt   # List of required dependencies
```

## Key Components
### PDF Text Extraction
The function `get_pdf_text` extracts text from all uploaded PDF files using `PyPDF2`.

### Text Chunking
The function `get_text_chunks` splits the extracted text into chunks of manageable sizes to optimize processing and retrieval.

### Vector Store Creation
The function `get_vectorstore` generates a vector store using Sentence Transformer embeddings and FAISS for efficient text search and retrieval.

### Conversational Chain
The `get_conversation_chain` function initializes the conversational pipeline using a pre-trained language model (`google/flan-t5-large`) from Hugging Face.

### Chat Interface
Custom HTML and CSS templates are used to style the chat interface for a better user experience.

## Limitations
- The application relies on the deprecated functions `create_retrieval_chain` and `HuggingFaceHub`.
- Text extraction from PDFs may not be perfect, especially for scanned documents or those with complex layouts.
- Hugging Face Hub's API usage might incur costs; ensure you monitor your API usage.

## Future Improvements
- Replace deprecated functions with up-to-date alternatives for better stability.
- Add support for other document types, such as Word or Excel files.
- Enhance the chat interface with advanced styling and user feedback mechanisms.
- Implement support for handling scanned PDFs using OCR.

## Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests with enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more information.


