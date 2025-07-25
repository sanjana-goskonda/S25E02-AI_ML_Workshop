!pip install llama-index
!pip install llama-index-embeddings-huggingface
!pip install peft
!pip install auto-gptq
!pip install optimum
!pip install bitsandbytes
!pip install gradio
!pip install pymupdf

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

Settings.embed_model = HuggingFaceEmbedding(model_name= "BAAI/bge-small-en-v1.5")
Settings.llm = None
Settings.chunk_size = 256
Settings.chunk_overlap = 25

pdf_loader = PyMuPDFReader()
pages = pdf_loader.load_data("/content/BotGenesis/MATRIX_CRITICAL_REVIEW_cinema&philosophy_SE22UCSE106.pdf")

print(len(pages))

index = VectorStoreIndex.from_documents(pages)

top_k = 13

retriever = VectorIndexRetriever(
    index = index,
    similarity_top_k= top_k,
)

query_engine = RetrieverQueryEngine(
    retriever= retriever,
    node_postprocessors= [SimilarityPostprocessor(similarity_cutoff=0.5)]
)

query = "Who embodies the quote IGNORANCE IS BLISS"
response = query_engine.query(query)

context = "Context: \n"
for i in range(top_k):
  context = context + response.source_nodes[i].text + "\n\n"
print(context)

print(len(response.source_nodes)) 

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

config = PeftConfig.from_pretrained("shawhin/shawgpt-ft")
model = PeftModel.from_pretrained(model, "shawhin/shawgpt-ft")

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

intstructions_string = f"""EnigmaGPT, functioning as a chatbot to answer questions on your personalized text material, communicates in clear, accessible language, escalating to technical depth upon request. \
It reacts to feedback aptly and ends responses with its signature '–EnigmaGPT'. \
EnigmaGPT will tailor the length of its responses to match the viewer's comment, providing concise acknowledgments to brief expressions of gratitude or feedback, \
thus keeping the interaction natural and engaging.

Please respond to the following comment.
"""
prompt_template = lambda comment: f'''[INST] {intstructions_string} \n{comment} \n[/INST]'''

comment = "Who embodies the quote IGNORANCE IS BLISS?" 

prompt = prompt_template(comment)
print(prompt)

tokenizer.pad_token = tokenizer.eos_token  
model.config.pad_token_id = tokenizer.pad_token_id

model.eval()  

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512  
)

outputs = model.generate(
    input_ids=inputs["input_ids"].to("cuda"),
    attention_mask=inputs["attention_mask"].to("cuda"),
    max_new_tokens=280
)

print(tokenizer.batch_decode(outputs)[0])

prompt_template_w_context = lambda context, comment: f"""[INST]EnigmaGPT, functioning as a chatbot to answer questions on your personalized text material, communicates in clear, accessible language, escalating to technical depth upon request. \
It reacts to feedback aptly and ends responses with its signature '–EnigmaGPT'. \
EnigmaGPT will tailor the length of its responses to match the viewer's comment, providing concise acknowledgments to brief expressions of gratitude or feedback, \
thus keeping the interaction natural and engaging.

{context}
Please respond to the following comment. Use the context above if it is helpful.

{comment}
[/INST]
"""

prompt = prompt_template_w_context(context, comment)

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=280)

print(tokenizer.batch_decode(outputs)[0])

import gradio as gr

pages = pdf_loader.load_data("/content/BotGenesis/MATRIX_CRITICAL_REVIEW_cinema&philosophy_SE22UCSE106.pdf")
index = VectorStoreIndex.from_documents(pages)
retriever = VectorIndexRetriever(index=index, similarity_top_k=1)

model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

def chatbot_response(query):
    """Retrieves context and generates an AI response."""
    response = retriever.retrieve(query)
    context = "\n".join([node.text for node in response])

    prompt = f"""[INST]EnigmaGPT, your AI study buddy, will answer based on the provided context:
{context}
Question: {query} [/INST]"""

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=280)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

demo = gr.Interface(
    fn=chatbot_response,
    inputs=gr.Textbox(lines=2, placeholder="Ask me a question..."),
    outputs=gr.Textbox(),
    title="EnigmaBot",
    description="A Retrieval-Augmented Chatbot to answer questions based on given data."
)

demo.launch(share=True)
