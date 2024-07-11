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
st.info("Welcome to BudgetBuddy! Ask me anything about the latest budget and finance of Pakistan.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if query := st.chat_input("Enter a message..."):
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

st.caption("Made with ❤️ by Shaheer Jamal")

