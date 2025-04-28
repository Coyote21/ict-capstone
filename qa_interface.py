import streamlit as st
from transformers import pipeline

# Load the trained model
@st.cache_resource
def load_qa_model():
    return pipeline(
        'question-answering', 
        model='../distilled_model',    # Local model path
        tokenizer='../distilled_model' # Local tokenizer path
    )

def main():
    st.sidebar.title("ICT Capstone Project 53")
    st.sidebar.subheader("Semester 1, 2025")
    st.sidebar.markdown("""Team Members:
- Vanessa Nguyen
- Yiran Wan
- Parnamika Ahuja
- Daniel Neal""")
    
    col1, col2 = st.columns([0.2, 0.8], vertical_alignment="center", border=False)
    with col1:
        st.image("./static/robot.svg")

    with col2:
        st.title("Question Answering")
    
    st.subheader("Prototype Distilled BERT-based AI Model")
    
    # Input section
    context = st.text_area("Provide context:", height=200)
    question = st.text_input("Ask a question:")
    
    qa_pipeline = load_qa_model()
    
    if st.button("Get Answer") and context and question:
        result = qa_pipeline({
            'context': context,
            'question': question
        })
        
        st.markdown("### Answer:")
        st.write(result['answer'])

if __name__ == "__main__":
    main()
