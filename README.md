# Medical Chatbot  

An intelligent, interactive medical chatbot designed to assist users with health-related queries. This project leverages cutting-edge tools and technologies, including LangChain for conversational AI, FastAPI for backend development, and Streamlit for deployment, ensuring an intuitive and seamless user experience.  

---

## Features  
- **Natural Language Understanding**: Understands user queries in natural language, ensuring smooth interaction.  
- **Interactive Interface**: A user-friendly interface built using Streamlit for seamless communication.  
- **Health-Related Insights**: Provides reliable and insightful responses to medical queries.  
- **Scalable Backend**: A robust backend powered by FastAPI ensures efficiency and scalability.  
- **Custom Knowledge Base**: Enhanced responses through integration with curated medical datasets.  

---

## Technologies Used  
- **LangChain**: Framework for building conversational AI with modular components.  
- **FastAPI**: Fast and efficient backend framework for handling API requests.  
- **Streamlit**: Simplified deployment with a clean and interactive user interface.  
- **FAISS**: Efficient vector search for quick and accurate document retrieval.  
- **Docker**: Containerization for easy deployment and scalability.  

---

## Setup and Installation  

### Prerequisites  
- Python 3.9 or above  
- Docker (optional, for containerized deployment)  

### Steps  

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/medical_chatbot.git  
   cd medical_chatbot  
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
3. **Run the Backend**
4. **Run the Frontend**
   ```bash
   streamlit run app.py
5. **Access the Chatbot**
   Open your browser and navigate


###Usage
1. **Launch the Streamlit application.**
2. **Enter your medical query into the chatbot interface.**
3. **Receive instant, AI-powered responses to your health-related questions.**

###Deployment with Docker
1. **Build the Docker Image**
   ```bash
   docker build -t medical_chatbot .  
2. **Run the Docker Container**
   ```bash
   docker run -p 8501:8501 medical_chatbot

###Project Highlights
1. Custom Model Deployment: Integrates a tailored AI model for domain-specific responses.
2. Efficient Search Mechanism: Utilizes FAISS for fast and accurate query handling.
3. Scalable Design: Built with components that allow easy scaling and feature expansion.


