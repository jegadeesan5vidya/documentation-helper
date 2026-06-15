from typing import Dict, Any, List
import streamlit as st

# Import the run_llm from our backend
from backend.core import run_llm

#define a function to format sources input: list of context langchain docs,
# output: list of URLs - to nicely render the output of URLs after we got results from the LLM
# def _format_sources(context_doc: List[Any]) -> List[str]:
#     return [
#         str(meta.get("source" or "unknown"))
#         for doc in (context_doc or [])
#         if (meta:= (getattr(doc, "metadata", None) or {})) is not None
#     ]

# Define above function in clean and readable version
def _format_sources(context_docs: List[Any]) -> List[str]:
    sources = []
    for doc in context_docs or []:
        meta = getattr(doc, "metadata", {}) or {}
        src = meta.get("source", "unknown")
        sources.append(str(src))
    # Remove duplicates while preserving order
    return list(dict.fromkeys(sources))

# StreamLit codes
st.set_page_config(page_title="LangChain Documentation Helper", layout="centered")

# NOTE: To run this source code use "pipenv run streamlit run main.py" in the terminal
# This will run streamlit application and use main.py as its source
st.title("LangChain Documentation Helper")
with st.sidebar:
    st.subheader("Session")
    # use_container_width=True means set button's width of its container which is the sidebar here
    if(st.button("Clear Chat", use_container_width=True)):
        #st.session_state["session"].clear() # OR
        st.session_state.pop("messages", None)
        # Rerun streamLit application
        st.rerun()
    # Display All the chat messages between the user and the LLM, for this iterate over
    # the messages and print them in the browser window
    # First display some welcome/placeholder messages, Note: Session State is a dictionary

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Ask anything about LangChain Documentation, I'll retrieve relevant context and cite sources.",
            "sources": [],
        }
    ]

# Iterate the "messages" from st.session_state and display its content in a container
# using chat_message, Note: The message could be user message, or AI message or String
# The chat_message will have a default Avatar, which will be changed bsed on the message's role
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)
        if message.get("sources"):
            # Display the sources in expander - multi element container dropdown which can be expanded/collapsed
            with st.expander("Sources"):
                for s in message["sources"]:
                    # Here - refers to show all the sources in list format
                    st.markdown(f"- {s}", unsafe_allow_html=True)

# Now define text area (a container) where user can input their message
# with a place holder "Ask a question...."
prompt = st.chat_input("Ask a question about LangChain...")
# If user entered any text, add that message into session's message and display it
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    # Since the prompt comes from user, we display the user avatar for this message
    with st.chat_message("user"):
        st.markdown(f"{prompt}", unsafe_allow_html=True)
    # We invoke our RAG LLM agent, and display a safe message if RAG fails, to avoid application crash
    with st.chat_message("assistant"):
        try:
            # Use st's spinner object
            with st.spinner("Retrieving docs and generating an answer..."):
                result: Dict[str, Any] = run_llm(str(prompt))
                answer = str(result.get("answer", "")).strip() or "(No answer returned.)"
                # Now get the formatted sources from context docs as well
                sources = _format_sources(result.get("context", []))
            st.markdown(answer, unsafe_allow_html=True)
            # Display the sources in expander - multi element container dropdown which can be expanded/collapsed
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"- {s}", unsafe_allow_html=True)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources,}
            )

        except Exception as e:
            st.error("Failed to generate a response from Assistant")
            st.exception(e)



