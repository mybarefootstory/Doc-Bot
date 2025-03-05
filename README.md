# Doc-Bot

Doc-Bot is an intelligent document-based chatbot designed to provide accurate and context-aware responses to user queries, with a special focus on medical literature. By leveraging advanced language models and vector search, Doc-Bot aims to assist healthcare professionals, students, and researchers by offering quick access to information from medical books and documents.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Features](#features)
- [Future Scope](#future-scope)
- [Contributing](#contributing)
- [Contact](#contact)

## Installation

To set up the Doc-Bot project, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/mybarefootstory/doc-bot.git
   cd doc-bot
   ```

2. **Set Up a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   Install the required packages using `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables:**
   Create a `.env` file in the root directory and add your Hugging Face token:
   ```plaintext
   HF_TOKEN="your_huggingface_token_here"
   ```

## Usage

1. **Create Memory for LLM:**
   Run the `create_memory_for_llm.py` script to process PDF documents and create vector embeddings:
   ```bash
   python create_memory_for_llm.py
   ```

2. **Connect Memory with LLM:**
   Run the `connect_memory_with_llm.py` script to connect the vector store with the language model:
   ```bash
   python connect_memory_with_llm.py
   ```

3. **Run the Chatbot:**
   Launch the chatbot using Streamlit:
   ```bash
   streamlit run docbot.py
   ```

   This will start a local server where you can interact with Doc-Bot through a web interface.

## Project Structure

- **create_memory_for_llm.py:** Loads PDF documents, creates text chunks, generates vector embeddings, and stores them in a FAISS vector store.
- **connect_memory_with_llm.py:** Similar to `create_memory_for_llm.py`, used for connecting the memory with the language model.
- **docbot.py:** The main script for running the Streamlit-based chatbot interface.
- **requirements.txt:** Lists all the dependencies required for the project.
- **.env:** Contains environment variables, such as the Hugging Face token.
- **.gitignore:** Specifies files and directories to be ignored by Git, such as `venv/` and `.env`.

## Features

- **Medical Document Processing:** Efficiently loads and processes medical PDF documents to provide quick access to information.
- **Vector Embeddings:** Utilizes Hugging Face models to create dense vector embeddings for semantic search.
- **Retrieval-Based QA:** Uses FAISS for fast similarity search and retrieval of relevant document sections.
- **Interactive Chatbot Interface:** Provides a user-friendly interface for interacting with the chatbot.

## Future Scope

- **Integration with Medical Databases:** Expand the database to include a wider range of medical literature and journals.
- **Enhanced NLP Capabilities:** Improve the natural language processing capabilities to better understand complex medical queries.
- **Multilingual Support:** Add support for multiple languages to cater to a global audience.
- **Mobile Application:** Develop a mobile app version to make the chatbot more accessible to healthcare professionals on the go.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## Contact

For questions or support, please contact [akashkumarswarnkar7172@gmail.com](mailto:akashkumarswarnkar7172@gmail.com).

```
