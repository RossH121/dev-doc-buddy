# Dev Doc Buddy

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chatbot-template.streamlit.app/)

Dev Doc Buddy is a Streamlit application that allows users to scrape, manage, and interact with development documentation using AI-powered assistance. It leverages Pinecone for vector storage and retrieval, and integrates multiple AI models for enhanced functionality.

## Project Structure
The project has the following file structure:

dev-doc-buddy/ 
├── .devcontainer/ 
│ └── devcontainer.json 
├── .github/ 
├── .streamlit/ 
│ └── secrets.toml 
├── pages/ 
│ ├── pycache/ 
│ ├── 1_Scrape_Docs.py 
│ ├── 2_Manage_Database.py 
│ └── 3_AI_Assistant.py 
├── utils/ 
│ ├── pycache/ 
│ ├── pinecone_assistant_utils.py 
│ └── pinecone_utils.py 
├── .gitignore 
├── LICENSE 
├── README.md 
├── requirements.txt 
└── streamlit_app.py

- `.devcontainer/`: Contains configuration for development containers.
- `.streamlit/`: Contains Streamlit configuration, including `secrets.toml` for storing sensitive information.
- `pages/`: Contains the individual Streamlit pages for different functionalities:
  - `1_Scrape_Docs.py`: Handles document scraping functionality.
  - `2_Manage_Database.py`: Manages the Pinecone database operations.
  - `3_AI_Assistant.py`: Implements the AI assistant chat interface.
- `utils/`: Contains utility functions and helpers:
  - `pinecone_assistant_utils.py`: Utility functions for Pinecone assistant operations.
  - `pinecone_utils.py`: General utility functions for Pinecone operations.
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `LICENSE`: Contains the project's license information.
- `README.md`: This file, providing an overview and instructions for the project.
- `requirements.txt`: Lists the Python dependencies required for the project.
- `streamlit_app.py`: The main Streamlit application file.

## Features

- **Document Scraping**: Scrape development documentation from websites and clean the content using AI.
- **Document Management**: Store and manage scraped documentation in a Pinecone vector database.
- **AI-Powered Chat Interface**: Interact with the documentation using multiple AI models, including:
  - GPT-4o mini
  - Claude 3.5 Sonnet
  - Pinecone Assistant
- **Token Usage Tracking**: Monitor and display token usage for context and output.
- **Database Management**: View, add, delete, and reprocess documents in the Pinecone database.

## Prerequisites

- Python 3.7+
- Streamlit
- Pinecone
- OpenAI API key
- Anthropic API key
- Google AI API key
- ScrapingAnt API key

## Installation

1. Clone the repository: git clone https://github.com/yourusername/dev-doc-buddy.git cd dev-doc-buddy

2. Install the required packages: pip install -r requirements.txt

3. Set up your environment variables or Streamlit secrets with the following keys:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `PINECONE_API_KEY`
- `SCRAPINGANT_API_KEY`

## Running the Application

To run the Dev Doc Buddy application: streamlit run streamlit_app.py

The application will be accessible at `http://localhost:8501` by default.

## Usage

1. **Home Page**: Provides an overview of the application's features.

2. **Scrape Docs**: 
   - Enter a URL to scrape development documentation.
   - View and edit the scraped content.
   - Save the content to Pinecone Index or as a Pinecone Assistant File.

3. **Manage Database**:
   - View database statistics.
   - Add test data to the database.
   - View stored data.
   - Delete specific data or clear all data.
   - Reprocess all documents in the database.

4. **Chat Interface**:
   - Select an AI model (GPT-4o mini, Claude 3.5 Sonnet, or Pinecone Assistant).
   - Interact with the AI using the scraped documentation as context.
   - View token usage statistics in the sidebar.

## Contributing

Contributions to Dev Doc Buddy are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Pinecone](https://www.pinecone.io/)
- [OpenAI](https://openai.com/)
- [Anthropic](https://www.anthropic.com/)
- [Google AI](https://ai.google/)
- [ScrapingAnt](https://scrapingant.com/)