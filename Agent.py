import torch
import pandas as pd
import numpy as np
from huggingface_hub import InferenceClient, login
from sentence_transformers import util, SentenceTransformer
from dotenv import load_dotenv, find_dotenv
from time import perf_counter as timer
import os
import streamlit as st

# Load the environment variables
load_dotenv(find_dotenv())
HF_TOKEN = os.getenv("hf_key") or st.secrets("HF_KEY")

model = InferenceClient(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    token=HF_TOKEN,
)

# import texts and embeddings df
text_and_embeddings_df = pd.read_csv("text_chunks_and_embeddings_df.csv")

# convert embedding column back to numpy array. (it got converted to string when it saved to CSV)
text_and_embeddings_df["embedding"] = text_and_embeddings_df["embedding"].apply(lambda x:np.fromstring(x.strip("[]"), sep=" "))

# Convert embeddings into torch.tensor
embeddings = torch.tensor(np.stack(text_and_embeddings_df["embedding"].to_list(), axis=0), dtype=torch.float32)

# convert texts into embedding df to list of dicts
pages_and_chunks = text_and_embeddings_df.to_dict(orient="records")

embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2",
                                     device="cpu")

def retrieve_relevant_resources(query: str,
                                embeddings : torch.tensor,
                                model: SentenceTransformer=embedding_model,
                                n_resources_to_return : int=5,
                                print_time: bool=True):
    """
    Embeds a query with model and returns top k scores and indices from embeddings.
    """
    # Embed the query
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Get dot product scores on embeddings
    start_time = timer()
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    end_time  = timer()
    
    if print_time:
        print(f"[INFO] Time taken to get scores on ({len(embeddings)}) embeddings : {end_time-start_time:.5f} seconds.")
        
    scores, indices = torch.topk(input=dot_scores,
                                 k=n_resources_to_return)
    
    return scores, indices

def prompt_formatter(query: str,
                     context_items: list[dict]) -> str:
    
    context= "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])
    base_prompt = """Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following examples as reference for the ideal answer style.
Dont say about the document. Just give the answer using the given context and your own.
And if you get asked anything that is not related to the budget, finances or tax, just say plesae ask questions about the budget, finances or tax of pakistan.
\nExample 1:
Query: What changes have been made to the income tax slabs and rates for individuals?
Answer: The Change in this budget for income tax slabs for salaried individuals are:
For taxable income up to Rs. 600,000 per annum the existing rate and the proposed rate are both 0%.
For taxable income from Rs. 600,001 per annum to Rs. 1,200,000, the proposed rate is 5% of the amount exceeding Rs. 600,000.
For taxable income from Rs. 1,200,001 per annum to Rs. 2,200,000, the proposed rate is Rs. 30,000 plus 15% of the amount exceeding Rs. 1,200,000.

\nExample 2:
Query: How this budget would impact it professionals working in the IT sector?
Answer: There is no direct mention of the impact of the budget on IT professionals working in the IT sector. The budget brief primarily focuses on tax proposals, sales tax rates, and fiscal policies, which may not have a direct impact on IT professionals.
However, some general insights can be drawn:
* The proposal to restrict foreign travel for non-filing of return, as well as the proposal to disallow 25% of sales promotion, advertisement, and publicity expense if a deduction has been claimed on account of royalty paid or payable to an associate, might affect IT professionals who frequently travel abroad for work or have international collaborations.
* The proposal to enhance the scope of "tax fraud" and introduce specific instances that may constitute tax fraud could lead to increased scrutiny of IT professionals' tax filings and potentially affect their tax compliance. 
* The proposal to introduce an electronic invoicing system might require IT professionals to adapt to a new system and potentially impact their workflow.

It is essential to note that these points are speculative and based on general insights, as there is no specific mention of IT professionals in the budget brief. A more detailed analysis or consultation with a tax expert would be required to assess the potential impact of the budget on IT professionals working in the IT sector.

\nExample 3:
Query: I am a student and I want to know how this budget would impact me?
Answer: As a student, you are not directly impacted by the tax rates or withholding tax rates mentioned in the budget brief. However, the budget does propose some changes that may indirectly affect you:
* The proposal to withdraw the exemption on income from subsidy granted by the Federal Government may impact your education expenses, if you receive any subsidies or scholarships.
* The extension of the exemption up to 30 June 2025 for residents of former FATA & PATA may benefit you if you are a student from these areas and receive any education-related benefits.
* The proposal to introduce electronic invoicing system may lead to changes in the way educational institutions and suppliers of educational materials operate, which may affect you indirectly.
It's worth noting that the budget does not propose any significant changes that would directly affect students. However, it's always a good idea to stay informed about the budget and its implications to ensure that you are aware of any changes that may affect your education or personal finances.
\nNow use the following context items to answer the user query:
{context}
User query: {query}"""

    base_prompt = base_prompt.format(context=context,query=query)
    
    return base_prompt

def ask(query: str,
        temprature:float=0.5,
        max_new_tokens: int=256,
        format_answer_text=True,
        return_answer_only=True):
    """
    Takes a query, finds relevant resources/context and generates an answer to the query to the query based on the relevant resources.
    """
    
    # Get the scores and indices of top related results
    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings)
    
    # Create a list of context items
    context_items = [pages_and_chunks[i] for i in indices]
    
    # Add score to context item
    for i, item in enumerate(context_items):
        item['score'] = scores[i].cpu()
        
    # AUGMENTATION
    # Create the prompt and format it with context items
    prompt = prompt_formatter(query=query,
                              context_items=context_items)
    
    # GENERATION
    response = model.chat_completion(
	messages=[
        {"role": "system", "content": "You are a budget expert who is answering queries based on the context items provided."},
        {"role": "user", "content": prompt}],
	max_tokens=max_new_tokens,
    temperature=temprature
    )
    
    # Extract the answer from the response
    for message in response['choices']:
        output = message['message']['content']
    output_text = ''.join(output)
    
    # Replace special tokens and unnecessary help message
    if format_answer_text:
        output_text = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "").replace("Sure, here is the answer to the user query:\n\n", "")
    
    # Only return the answer without the context items
    if return_answer_only:
        return output_text

    return output_text, context_items

def main(query:str):

    login(token=HF_TOKEN, add_to_git_credential=True)
    answer = ask(query=query)
    return answer