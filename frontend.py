import streamlit as st
from prompts import meta_prompt
import openai 
import dotenv 
import os 
import re 
import json 
import pymupdf4llm

dotenv.load_dotenv(override=True)

st.set_page_config(layout="wide") 
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stToolbar"] {display: none !important;}
    [data-testid="stSidebarNav"] {display: none !important;}
    </style>
    """,
    unsafe_allow_html=True
)

model: str = "gpt-4o-mini"
OPENAI_API_KEY : str = os.getenv("OPENAI_API_KEY") 

meta_prompt: str = """
You are a prompt generation bot. Your task is to read the user's instruction and generate an ENGINEERED PROMPT that is structured for subsequent language model processing.

### DEFINITIONS ###
- VAGUE REQUEST: An imprecise or unstructured demand from the user that lacks specific formatting or detailed instructions.
- Task: A clearly defined objective that must be achieved. Rephrase the user's idea into a precise, technical task statement.
- Inputs: The data provided by the user that is necessary to complete the task. Each input should be labeled as [INPUT VALUE N] and later replaced by its name (without brackets) followed by empty curly braces (e.g. `[curly-braces]`) to allow Python f-string formatting.
- Output: The final result demonstrating that the objective has been met.
- Expert-title: A creative, domain-specific title that establishes the LLM as an expert in the relevant field.

### ENGINEERED PROMPT FORMAT ###
```
You are a [expert-title]. Your goal is to [task].

### DEFINITIONS ###
[Define all relevant terms needed for the task.]

### INSTRUCTIONS ###
[Break down the task into a clear sequence of steps for the LLM.]

### OUTPUT FORMAT ### 
* Your output should be enclosed within <output></output> tags. 
* Within the output tag should be a stringifiable JSON dictionary. 
// additional output details of how the json should be structured. 

```

### INSTRUCTIONS ###
1. Replace `expert-title` with an imaginative title that positions the LLM as a subject matter expert.
2. Rephrase the user's vague request into a clearly defined technical task.
3. In the DEFINITIONS section, explain any key terms that the LLM must understand.
4. Break the task into detailed, step-by-step instructions in the INSTRUCTIONS section.

### USER TASK ###
{}

### OUTPUT THE PROMPT SHOULD PROVIDE ### 
The output should always be in JSON dictionary.  
"""

predefined_prompt_hashset: str = {
  "AI Risks Identification Prompt": "What risks are associated with using a generative AI tool to autonomously (a) review accommodations applications; (b) approve or deny requests; and (c) send out notice of decisions to candidates (and if the request is granted, to also send the information to delivery vendors)?",
  "AI Risk Mitigation Prompt": "What are the best ways to mitigate those risks?"
}  

client = openai.OpenAI(api_key=OPENAI_API_KEY)

def extract_markdown_per_page(pdf_path):
    page_chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    markdown_list = [page_chunk['text'] for page_chunk in page_chunks]
    return markdown_list

def rag_page():
    st.markdown("<hr><h3 style='text-align: center; font-size: 25px; margin-top: 10px; margin-bottom: -5px;'>Predefined Prompts</h3>", unsafe_allow_html=True)
    dropdown_container, prompt_output_container = st.columns(2) 

    with dropdown_container: 
        st.markdown("<h4 style='text-align: center; font-size: 18px;'>Predefined prompt Options</h4>", unsafe_allow_html=True)
        model = st.selectbox("Select an OpenAI model: ", ("gpt-4o-mini", "gpt-4o"))
        st.session_state.model = model
        selected_option = st.selectbox("Select a particular prompt: ", list(predefined_prompt_hashset.keys())) 
        prompt: str = predefined_prompt_hashset.get(selected_option)
        st.code(prompt, height=150, language="markdown", wrap_lines=True)
        btn = st.button("Generate!")
            

    with prompt_output_container: 
        st.markdown("<h4 style='text-align: center; font-size: 18px;'>Predefined prompt Output displayed here</h4>", unsafe_allow_html=True)
        if selected_option and btn: 
            with st.spinner("Generating output..."): 
                prompt: str = predefined_prompt_hashset.get(selected_option)
                add_text_prompt = (prompt + "### TEXT CONTENT ###\n" +  
                    ".".join(st.session_state.markdown_pages) ) 
                
                response = client.chat.completions.create(
                    model=st.session_state.model, 
                    messages=[{
                        "role": "user", 
                        "content": add_text_prompt, 
                    }], 
                ) 

                st.session_state.predefined_prompt_response = response.choices[0].message.content
                st.code(st.session_state.predefined_prompt_response, height=500, wrap_lines=True, language="markdown")  
        else: 
            st.code("Predefined prompt output will appear here...", height=500, wrap_lines=True, language="markdown")

    st.markdown("""
                <hr>
                <h3 
                    style='text-align: center; 
                    font-size: 25px; 
                    margin-top: 10px; 
                    margin-bottom: 2px;'
                >Meta Prompting
                </h3>""", 
                unsafe_allow_html=True)
    st.markdown("""<p 
                    style='
                    text-align: center; 
                    color: green; 
                    background-color: #21252d; 
                    border-radius: 20px; 
                    '>Note: The model selected above is also reused here below
                </p>""", 
                    unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h5 style='text-align: center;'>User prompt</h5>", unsafe_allow_html=True)
        st.markdown("Enter any prompt you need into the `user prompt box` box to interact and extract information from the pdf. The output of your prompt will be provided below. This prompt will also be used by the system as a reference to generate a more refined version of itself.")
        normal_prompt = st.text_area("user prompt box", height=800, placeholder="Enter your prompt here...")

        outputBtn = st.button("Generate Outputs")
    
    if normal_prompt and outputBtn:
        engineered_prompt = meta_prompt.format(normal_prompt)  
        response = client.chat.completions.create(
            model=st.session_state.model, 
            messages=[{
                "role": "user", 
                "content": engineered_prompt, 
            }], 
        ) 
        content = response.choices[0].message.content  
        if content.startswith("```"): 
            content = content[3:]
        
        if content.endswith('```'): 
            content = content[:-3]

        st.session_state.engineered_prompt = content  

    with col2:
        st.markdown("<h5 style='text-align: center;'>Engineered prompt</h5>", unsafe_allow_html=True)
        st.markdown("This textbox displays an Engineered prompt in the `engineered prompt box` generated using your input as reference. The Engineered prompt is a special type of prompt which ensures structure and better quality of outputs.")
        st.markdown("<p style='margin-bottom: 5px;'>engineered prompt box</p>", unsafe_allow_html=True)
        engineered_prompt = st.code(st.session_state.engineered_prompt, 
                                         height=800, 
                                         language="markdown", 
                                         wrap_lines=True)

    st.markdown("<h3 style='text-align: center; font-size: 25px;'> Generate & Compare!</h3>", unsafe_allow_html=True) 
    out_col1, out_col2 = st.columns(2)
    with out_col1:
        st.markdown("<h5 style='text-align: center;'>Output for User Prompt:</h5>", unsafe_allow_html=True)
        if normal_prompt and outputBtn:
            st.write(f"Processed output for: {normal_prompt}")
            with st.spinner("Normal text output..."): 
                if "markdown_pages" in st.session_state: 

                    normal_prompt = (normal_prompt + "### TEXT CONTENT ###\n" +  
                        ".".join(st.session_state.markdown_pages) ) 

                    response = client.chat.completions.create(
                        model=st.session_state.model, 
                        temperature=0.1, 
                        messages=[{
                            "role": "user", 
                            "content": normal_prompt, 
                        }], 
                    ) 

                    st.markdown(response.choices[0].message.content) 

        else:
            st.write("No user prompt provided.")

    with out_col2:
        st.markdown("<h5 style='text-align: center;'>Output for Engineered Prompt:</h5>", unsafe_allow_html=True)
        if st.session_state.engineered_prompt != "Engineered prompt will appear here...": 
            with st.spinner("Output for Engineered prompt..."):
                actual_prompt_to_send = st.session_state.engineered_prompt
                
                if "markdown_pages" in st.session_state:
                    actual_prompt_to_send += "### PDF CONTENT###\n" + ".".join(st.session_state.markdown_pages)
                
                response = client.chat.completions.create(
                    model=st.session_state.model, 
                    temperature=0.1, 
                    messages=[{
                        "role": "user", 
                        "content": actual_prompt_to_send, 
                    }], 
                )
                
                try:
                    response_content = response.choices[0].message.content
                    if re.findall(r"<output>(.*?)</output>", response_content, re.DOTALL):
                        json_content = re.findall(r"<output>(.*?)</output>", response_content, re.DOTALL)[0]
                        if json_content.startswith("```json"):
                            json_content = json_content.split("```json")[1].split("```")[0]
                        js_dict = json.loads(json_content)
                        st.json(js_dict)
                    else:
                        st.code(response_content)
                except Exception as e:
                    st.markdown(response.choices[0].message.content)
                    print(f"Error processing response: {e}")

def main():
    
    if ("engineered_prompt" not in st.session_state) : 
        st.session_state.engineered_prompt = "Engineered prompt will appear here..."

    if ("markdown_pages" not in st.session_state) : 
        pdf_path: str = "./assets/acceptable-policies.pdf" 
        st.session_state.markdown_pages = extract_markdown_per_page(pdf_path)
    
    if ("model" not in st.session_state): 
        st.session_state.model = "gpt-4o-mini"
    
    oess, title, atp = st.columns([1, 8, 1])
    
    with title:  
        st.markdown(
            "<h1 style='text-align: center; font-size: 35px; "
            "margin-top: -20px; '>"
            "Workshop - Practical, Legal and Ethical considerations</h1>",
            unsafe_allow_html=True
        )

        st.markdown(
            "<h2 style='text-align: center; font-size: 25px; "
            "margin-top: -20px; margin-bottom: 30px;'>"
            "While navigating the roadmap of Generative AI adoption</h2>",
            unsafe_allow_html=True
        )

    url = "https://atp2025.theopeneyes.com/sample/ATP2025-GenAIAcceptableUSPolicySample.pdf"

    with oess: 
        st.image(
            "./assets/OET_Logo.png", 
            width=165, 
        )
    
    with atp: 
        st.image(
            "./assets/adenosine-tri-phosphate.jpg", 
            width=185
        )

    st.markdown(
        """
        <hr>
        <div style="text-align: center; margin-top: 10px; ">
            <h3 style="font-size: 25px;">About Policy Information used in this demo</h3>
            <p style="margin-left: 180px; margin-right: 180px; font-size: 15px; margin-bottom: 2px; ">The Policy provided below is a sample policy outlining guidelines for the safe and responsible use of AI technology in the workplace. </p>
            <p style="margin-left: 180px; margin-right: 180px; font-size: 15px; margin-bottom: 2px; ">It defines key terms, details security best practices, and explains staff responsibilities for using AI tools ethically and securely.</p>
        </div>
        """, 
        unsafe_allow_html=True
    ) 

    st.markdown(
        f"""
        <div style="text-align: center; margin-top: 20px;">
            <a href="{url}" target="_blank">
                <button style="font-size:16px; margin-top: -70px; margin-bottom: 30px; padding:10px 20px; cursor:pointer; border: solid black 3px; background-color: rgb(14, 205, 142); border-radius: 20px; ">
                    View Policy 
                </button>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    ) 
    rag_page()

if __name__ == "__main__":
    main()
