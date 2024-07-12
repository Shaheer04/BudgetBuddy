from Agent import main
import streamlit as st
import time

# Define a function to get response from the agent  
def agent_response(query:str):
    response = main(query=query)
    return response

# Define a generator function to display response word by word
def response_generator(text: str):
    for paragraph in text.split('\n'):
        if paragraph.strip():  # Only process non-empty paragraphs
            for word in paragraph.split():
                yield word + " "
                time.sleep(0.05)
            yield "\n\n"  # Add paragraph breaks

st.title("BudgetBuddy")
st.info("""
        I am your Budget Buddy, Ask me anything about the latest budget and finance of Pakistan.
        """)

def conversation(query:str):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)

    # Get assistant response and display it
    with st.chat_message("assistant"):
        response = agent_response(query)
        response = st.write_stream(response_generator(text=response))
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Enter a message..."):
    conversation(query=query)

if st.button("What is the property transfer tax in this budget?", use_container_width=True, type="primary"):
    conversation(query="What is the property transfer tax in this budget?")

if st.button(" What changes have been made to the income tax slabs and rates for individuals?",  use_container_width=True, type="primary"):
    conversation(query="What changes have been made to the income tax slabs and rates for individuals?")

if st.button("What is the rate of withholding tax on cash withdrawal from banks for non-filers?",  use_container_width=True, type="primary"):
    conversation(query="What is the rate of withholding tax on cash withdrawal from banks for non-filers?")

st.caption("Made with ❤️ by Shaheer Jamal")

