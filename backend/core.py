# Implementation for the entire RAG pipeline retrieval with and agent that has a
# retriever, this will retrieve relevant context from the document.  The retrieval
# will be marked as ToolMessage

import os
from typing import Any, Dict
from dotenv import load_dotenv
from langchain.agents import create_agent
# Quick way to create a chat client to make LLM request, this takes model
# as string and return us the correct model
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain_core.tools import tool
from langchain_pinecone import PineconeVectorStore
# To embed the user query into a vector before we get relevant document context from the store
from langchain_openai import OpenAIEmbeddings
from openai.types.shared_params import response_format_text

load_dotenv()

# Initialize embedding models
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", show_progress_bar=False, chunk_size=50, retry_min_seconds=10)

# Initialize vector store
vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)

# Initialize the chat model
model = init_chat_model("gpt-5.2", model_provider="openai")

# Define a Tool using the too decorator to retrieve the content for the given
# user's query, content_and_artifact is to configure the tool to return the response (i.e., ToolMessage)
# part 1 - The textual or structured response and part 2 -  A secondary output/artifact,
# something non-textual such as an image, file, or other generated object
# The part 1 of ToolMessage will be feed into downstream LLM and Part 2 will be
# for application logic (refer to ToolMessage structure pic from phone)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    # Now get top 4 most relevant documents from the vector store for the given user query
    # Here the invoke method is going to perform the similarity search and returns top 4
    # related contents (output will be list of langchain Document
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)

    # Next step is prompt augmentation, which requires serialization of all the
    # documents in one line.
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    # Return both serialized docs and retrieved raw docs
    # with this we are downstreaming this serialized docs (part1) to the LLMM
    # and send the raw retrieved_docs (par2) to the downstream appications/systems
    return serialized, retrieved_docs

def run_llm(query: str) -> Dict[str, Any]:
    """Run the RAG pipeline to answer a query using the retrieved documentation.
    Args:
        query (str): The user's query to run
    Returns:
        Dictionary containing:
            - answer: The generated answer
            - contexT: List of retrieved documents
    """
    # Create the agent with retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    # Build query message list
    messages = [{"role": "user", "content": query} ]

    # Now invoke the agent which expects a dictory that contains the messages
    response = agent.invoke({"messages": messages})

    # Extract the answer from last AI message, Meaning is “From the response, take the
    # list of messages, pick the last one, and store its content in answer.”
    # Why last message? - The last message is always the model’s final answer, this applies to all the models.
    answer = response["messages"][-1].content

    # Note: Creating user's trust in our model: Along with the final answer, the model
    # also sends the retrieved raw documents to share the sources for final answer
    # This is very critical for agentic user experience

    # Extract the context(artifact) documents from Tool Message
    context_docs=[]
    for message in response["messages"]:
        # Check if the ToolMessage contains content and artifacts
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)
    return {
        "answer": answer,
        "context": context_docs
    }

if __name__ == "__main__":
    result = run_llm("What are deep agents")
    print(result)

