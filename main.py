import torch
import pandas as pd
import numpy as np
from huggingface_hub import InferenceClient, login
from sentence_transformers import util, SentenceTransformer
from dotenv import load_dotenv, find_dotenv
from time import perf_counter as timer
import os

# Load the environment variables
load_dotenv(find_dotenv())
HF_TOKEN = os.getenv("hf_key")

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
\nExample 1:
Query: What changes have been made to the income tax slabs and rates for individuals?
Answer: The Finance Bill proposes to revise the tax rates for salaried individuals. The comparison of existing and proposed rates is as follows:
        For taxable income up to Rs. 600,000, the existing rate and the proposed rate are both 0%.
        For taxable income from Rs. 600,001 to Rs. 1,200,000, the existing rate is 2.5% of the amount exceeding Rs. 600,000, while the proposed rate is 5% of the amount exceeding Rs. 600,000.
        For taxable income from Rs. 1,200,001 to Rs. 2,200,000, the existing rate is Rs. 15,000 plus 12.5% of the amount exceeding Rs. 1,200,000, while the proposed rate is Rs. 30,000 plus 15% of the amount exceeding Rs. 1,200,000.
\nExample 2:
Query: What is the sales tax on mobile devices?
Answer: The Bill proposes to amend Table-II of the Ninth Schedule, which outlines the sales tax rates for cellular mobile phones or satellite phones based on import value per set or equivalent value in rupees for manufacturer supplies. The comparison between existing and proposed rates for each category is as follows:
        Category A: Not exceeding US$ 30 (excluding smartphones)
            Sales tax on CBUs at the time of import or registration (IMEI number by CMOs): Rs. 130 (Old) → 18% (New)
            Sales tax on import in CKD / SKD condition: Rs. 10 (Old) → 18% (New)
            Sales tax on supply of locally manufactured mobile phones in CBU condition in addition to tax under column (4): Rs. 10 (Old) → 18% (New)
        Category B: Not Exceeding US$ 30 (smartphones)
            Sales tax on CBUs at the time of import or registration (IMEI number by CMOs): Rs. 200 (Old) → 18% (New)
            Sales tax on import in CKD / SKD condition: Rs. 10 (Old) → 18% (New) 
\nExample 3:
Query: What modifications have been made to the GST rates for various goods and services?
Answer: The Bill proposes the following amendments
        Sales tax on supplies and imports of plant, machinery, and electricity in tribal areas is proposed to be charged at a reduced rate of 6% for FY 2025 and 12% for FY 2026.
        Sales tax at the rate of 10% is proposed on oil cake and other solid residues, and tractors by placing these items in the Eighth Schedule of the Act.
        Table II – Local supplies only:
            Sales tax at the rate of 10% is proposed on the local supply of vermicelli, sheer mal, bun, and rusk (excluding those sold in bakeries and sweet shops falling in the category of Tier-1 retailers) by placing these items in the Eighth Schedule of the Act.
            Sales tax at the rate of 10% is proposed on the local supply of poultry feed, cattle feed, sunflower seed meal, rape seed meal, and canola seed meal by placing these items in the Eighth Schedule of the Act, with the condition that refund of excess input tax, if any, shall not be admissible.
\nNow use the following context items to answer the user query:
{context}
User query: {query}
Answer:"""
    
    base_prompt = base_prompt.format(context=context,query=query)
    
    return base_prompt

def ask(query: str,
        temprature:float=0.7,
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
	messages=[{"role": "user", "content": prompt}],
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

login(token=HF_TOKEN, add_to_git_credential=True)
answer = ask("What will be the tax rates on the purchase of 1000sqft plot?")
print (answer)