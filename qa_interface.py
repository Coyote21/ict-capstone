import streamlit as st
from transformers import pipeline

# Load the trained model
@st.cache_resource
def load_qa_model():
    return pipeline(
        'question-answering', 
        model='./distilled_model',    # Local model path
        tokenizer='./distilled_model' # Local tokenizer path
    )

def main():
    st.title("Question Answering with Crafted Resoning's MiniRoBERTa-Edge")
    st.markdown("### Provide context and ask a question")
    
    # Input section
    context = st.text_area("Enter context:", height=200)
    question = st.text_input("Ask a question:")
    
    qa_pipeline = load_qa_model()
    
    if st.button("Get Answer") and context and question:
        result = qa_pipeline({
            'context': context,
            'question': question
        })
        
        st.markdown("### Answer:")
        st.markdown(f"**{result['answer']}**")
        st.write(f"Confidence score: {result['score']:.2%}")

if __name__ == "__main__":
    main()
