# 🧑‍🌾 Farmers AI

**Farmers AI** is an AI-powered assistant designed to help farmers by answering questions related to fertilizers, crops, and diseases. It also allows image uploads for plant diagnosis using advanced AI models.

---

## 🚀 Features

- 🌿 Ask farming-related questions
- 🧪 Get fertilizer recommendations
- 📷 Upload plant images for disease diagnosis
- 🧠 Powered by:
  - [Chainlit](https://www.chainlit.io/)
  - [Ollama](https://ollama.com/)
  - [LangChain](https://www.langchain.com/)

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/kushal867/farmers-ai.git
cd farmers-ai
2. Create Virtual Environment (Optional but recommended)
bash
Copy
Edit
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
# or, if using pyproject.toml:
pip install -e .
🖼️ Image Upload Functionality
Users can upload images of crops or leaves.

The model will process the image and return a diagnosis or suggestion.

🧪 Run the App
You likely have two key files: app.py and main.py.

To run the Chainlit app:

bash
Copy
Edit
chainlit run app.py
Or:

bash
Copy
Edit
python main.py
📂 Project Structure
css
Copy
Edit
farmers-ai/
├── app.py
├── main.py
├── chainlit.md
├── README.md
├── pyproject.toml
├── .gitignore
✨ Example Use Cases
“What fertilizer should I use for tomatoes?”

“Is this leaf infected?” (upload an image)

“How to improve soil pH?”

🤖 Requirements
Python 3.8+

Chainlit

LangChain

Ollama

OpenAI or other compatible LLM provider (optional)

📝 License
MIT License. Feel free to use and contribute!

🙌 Contributing
Pull requests are welcome. For major changes, please open an issue first.

