import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

@st.cache_resource
def load_model():
    """Load model and tokenizer with caching"""
    model_name = "Kan05/Paper-Title-Generator-GPT2"  # Your HuggingFace model path
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer

def generate_title(abstract, model, tokenizer, max_new_tokens=50):
    """Generate title from abstract"""
    input_text = f"Abstract: {abstract} Title:"
    
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Generate title
    output_sequences = model.generate(
        input_ids=inputs["input_ids"],
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    
    # Decode the generated title
    generated_title = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return generated_title.replace(input_text, "").strip()


# Set up the Streamlit interface
st.title("Research Paper Title Generator")
st.write("Generate titles from paper abstracts using GPT-2")

# Text area for single abstract input
single_abstract = st.text_area(
    "Enter your paper abstract:",
    height=200,
    help="Paste your research paper abstract here"
)

# File uploader for batch processing
uploaded_file = st.file_uploader(
    "Or upload a file with multiple abstracts (one per line):",
    type=['txt']
)

# Generation parameters
with st.expander("Advanced Settings"):
    max_length = st.slider(
        "Maximum title length",
        min_value=30,
        max_value=100,
        value=50
    )

if st.button("Generate Title(s)"):
    # Load model
    with st.spinner("Loading model..."):
        model, tokenizer = load_model()
    
    # Process single abstract
    if single_abstract:
        with st.spinner("Generating title..."):
            title = generate_title(single_abstract, model, tokenizer, max_length)
            st.subheader("Generated Title:")
            st.write(title)
    
    # Process uploaded file
    elif uploaded_file:
        abstracts = uploaded_file.getvalue().decode().split('\n')
        abstracts = [abs.strip() for abs in abstracts if abs.strip()]
        
        st.subheader("Generated Titles:")
        for idx, abstract in enumerate(abstracts, 1):
            with st.spinner(f"Generating title {idx}/{len(abstracts)}..."):
                title = generate_title(abstract, model, tokenizer, max_length)
                st.write(f"**Abstract {idx}:**")
                st.write(f"Title: {title}")
                st.write("---")
    else:
        st.warning("Please enter an abstract or upload a file.")

# Add instructions in sidebar
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Enter a single abstract in the text area, or
2. Upload a text file with multiple abstracts (one per line)
3. Adjust the maximum length setting if needed
4. Click 'Generate Title(s)' to get results

Note: First generation might take longer as the model loads.
""")

# Add example abstract
st.sidebar.header("Example Abstract")
example_abstract = """This paper discusses a novel approach to optimizing machine learning models through the use of hybrid techniques combining supervised and unsupervised learning."""
st.sidebar.write(example_abstract)