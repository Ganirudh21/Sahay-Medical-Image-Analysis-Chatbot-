import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from ultralytics import YOLO
import os

# Set environment variables (if necessary)
os.environ["LANGCHAIN_API_KEY"] = "your_langchain_api_key"

# Rich Medical Knowledge Base for Pneumonia (for the chatbot's follow-up responses)
medical_knowledge_base = {
    "pneumonia": {
        "description": "Pneumonia is an infection that inflames the air sacs in one or both lungs, which may fill with fluid or pus. Causes include bacteria, viruses, or fungi.",
        "symptoms": [
            "Fever, chills, or sweating",
            "Coughing that produces phlegm",
            "Shortness of breath",
            "Fatigue"
        ],
        "care_advice": [
            "Get plenty of rest to allow your body to recover.",
            "Stay hydrated by drinking water and warm fluids.",
            "Follow your doctor's prescribed antibiotic or antiviral medication plan.",
            "Use a humidifier to ease breathing.",
            "Avoid smoking and exposure to secondhand smoke."
        ],
        "mental_health_advice": "It's natural to feel worried when you're unwell. Rest your mind by engaging in calming activities like listening to soothing music or meditating. Remember, recovery takes time.",
        "references": [
            {
                "source_type": "Book",
                "title": "Davidson's Principles and Practice of Medicine",
                "excerpt": "Pneumonia often presents with acute symptoms such as fever, cough, and breathlessness, and may require antibiotic therapy for bacterial causes."
            },
            {
                "source_type": "Book",
                "title": "Harrison's Principles of Internal Medicine",
                "excerpt": "In pneumonia, bacterial or viral pathogens infiltrate alveoli, leading to inflammatory responses. Hydration and oxygen therapy are critical for management."
            },
            {
                "source_type": "Journal",
                "title": "The Lancet - Pneumonia in Adults",
                "excerpt": "Effective treatment of pneumonia involves pathogen-specific antibiotics and supportive care like fluids and oxygen therapy."
            },
        ]
    }
}

# Prompt Template for LLM (for chatbot interaction)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are Sahaay, a medical assistant providing detailed and scholarly responses. Cite multiple references to ensure credibility."),
        ("user", "{message}")
    ]
)

# Fetch Medical Info from Knowledge Base
def fetch_medical_info(query):
    query = query.lower()
    for key, value in medical_knowledge_base.items():
        if key in query:
            return value
    return None

# Generate Response (from Knowledge Base or LLM)
def generate_response(message, chat_history=[], llm_model="llama3.2", temperature=0.7, max_tokens=150):
    topic_info = fetch_medical_info(message)
    if topic_info:
        # Prepare a response from the knowledge base
        response = f"""
        ### 📚 Medical Information:
        **Description**: {topic_info['description']}
        
        **Symptoms**:
        - {', '.join(topic_info['symptoms'])}
        
        **Care Advice**:
        - {', '.join(topic_info['care_advice'])}
        
        **Mental Health Note**:
        {topic_info['mental_health_advice']}
        
        **References**:
        {"".join([f"- From *{ref['title']}*: {ref['excerpt']}\n" for ref in topic_info['references']])}
        """
        return response.strip()
    else:
        # Use LLM for non-knowledge-base queries
        llm = Ollama(model=llm_model)
        output_parser = StrOutputParser()
        chain = prompt_template | llm | output_parser
        context = "\n".join(chat_history + [message])
        answer = chain.invoke({'message': context})
        return answer.strip()

# Step 1: Load your trained model
model = YOLO('C:\\Users\\aniru\\Desktop\\abc\\best.pt')  # Replace with the actual path to your best.pt model

# Step 2: Function to run inference on the uploaded image
def classify_image(image_path):
    results = model(image_path)
    result = results[0]  # Get the result for the single image
    predicted_class_idx = result.probs.top1  # Index of the predicted class
    predicted_class = result.names[predicted_class_idx]  # Predicted class name
    confidence_score = result.probs.data[predicted_class_idx]  # Confidence score
    return predicted_class, confidence_score

# Streamlit App (Main)
def main():
    # Page Configuration
    st.set_page_config(page_title="Sahaay: Medical Chatbot", page_icon="🩺", layout="centered")
    st.title("🩺 Sahaay: Your Caring Medical Assistant")
    st.markdown(
        """
        Welcome to **Sahaay**! I provide medical information and support with insights from trusted references like books and journals.
        """
    )

    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Image Upload
    st.markdown("### 📸 Upload a Chest X-ray Image for Analysis")
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Save the uploaded image temporarily in the 'uploads' folder
        local_file_path = os.path.join("uploads", "uploaded_image.jpeg")
        
        # Make sure the directory exists
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Save the image to the local path
        with open(local_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Classify the Image using the trained model
        predicted_class, confidence_score = classify_image(local_file_path)

        # Show the prediction result
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.write(f"**Predicted Class:** {predicted_class}")
        st.write(f"**Confidence Score:** {confidence_score:.2f}")

        # Start chatbot conversation based on the result
        if predicted_class.lower() == "pneumonia":
            response = "The image suggests you may have pneumonia. Would you like to know more about it?"
        else:
            response = "The image suggests that your lungs appear normal. Let me know if you have any other questions."

        # Update Chat History and Display (Only show once per session)
        if not any(message.startswith("🤖 The image suggests") for message in st.session_state.messages):
            st.session_state.messages.append(f"🤖 {response}")

    # Display Chat History in Reverse Order (most recent at top)
    for message in reversed(st.session_state.messages):
        if message.startswith("👤"):
            st.write(message)
        else:
            st.markdown(message)

    # Input field for additional questions or interaction (fixed at the bottom)
    st.markdown("<div style='position:fixed;bottom:20px;width:100%;'>", unsafe_allow_html=True)
    user_input = st.text_input("Ask me anything related to your health or the image analysis", key="user_input_field")
    st.markdown("</div>", unsafe_allow_html=True)

    if user_input:
        # Append user input to chat history and generate response
        st.session_state.messages.append(f"👤 {user_input}")
        response = generate_response(user_input, chat_history=st.session_state.messages)
        st.session_state.messages.append(f"🤖 {response}")

        # Display updated chat history with new message at the top
        for message in reversed(st.session_state.messages):
            if message.startswith("👤"):
                st.write(message)
            else:
                st.markdown(message)

    # Sidebar Information
    st.sidebar.header("ℹ️ About Sahaay")
    st.sidebar.markdown(
        """
        **Sahaay** is designed to assist with medical information based on scholarly references.
        
        *Disclaimer*: This chatbot does not substitute professional medical advice. For serious concerns, consult a healthcare provider.
        """
    )

    # Footer
    st.markdown("---")
    st.markdown("💡 *Stay informed, stay healthy!*")

if __name__ == "__main__":
    main()

