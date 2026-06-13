import asyncio
import os
import ssl #Need SSL As we are going to create SSL context for some type hinting objects
from importlib.metadata import metadata
from typing import Any, Dict, List

# We need to import certified package to get us valid certificate to attach it to our
# http request
import certifi
from chromadb.utils import results
# Import dotenv to load environment variables
from dotenv import load_dotenv
from numba.core.cgutils import false_bit

load_dotenv()

# now import all langchain related libraries
# RecursiveCharacterTextSplitter is langchain helper class to split the documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Chroma is a vector store to index everything locally instead of cloud. Note: We will use
# pinecone which is cloud based
from langchain_chroma import Chroma
# Document class to handle text data which can be processed, split and embedded or indexed
from langchain_core.documents import Document
# Alternatively we can use any other open source embeddings like MistralAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
# Tavily is going to be the main driver to get langchain's documentation
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

# Now import all logging functions from logger.py
from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)

# Load environment variables
load_dotenv()

# Configure SSL context with a valid certificate using certifi package. This is for making
# tons of Tavily API requests
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
print(f"certifi.where() {certifi.where()}")
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# show_progress_bar as true, so that we can see a progress bar while text is being indexed into a vector
# RATE LIMITING: Also we set chunk_size to 50, to limit no. of documents that we can embed per request.
# Which means, we are going to embed 50 linkchain documents or text objects in a single request
# if chunk_size is too low, i.e., 1, embedding process takes lot longer to process & embed everything
# RETRY_MIN_SECONDS: in case of failures after long processes (hour long) for some reason (due to wrong payload etc.,),
# this will trigger rate limits checks.  One strategy to manage this scenario is, use this retry with 10 sec.
# This means in case of  a failure, we are going to wait and re-try to emebd text after 10 sec to avoid rate limits.
# NOTE: rate limiting errors and handling rate limits is very common when we take application to production
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", show_progress_bar=False, chunk_size=50, retry_min_seconds=10)

# Below chroma will create a db under current working directory of this project
# New folder called chroma_db will be created inside this project folder where
# the vector details will be stored locally using sqllite db, we can query this db and see output
# select * from embeddings
vectorStore=Chroma(persist_directory="chroma_db", embedding_function=embeddings)
#vectorStore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()
print("Initialisation of object activity is completed")

#DEFINE HELPER FUNCTIONS chunk_urls
def chunk_urls(urls: List[str], chunk_size: int=10) -> List[List[str]]:
    """Split the list of of URLs into chunks of specified size"""
    chunks = []
    for i in range (0, len(urls), chunk_size):
        chunks.append(urls[i:i+chunk_size])
    return chunks

# Below Coroutine Concurrent Extract batch, receives list of urls and batch number (for observability)
# Extract the document contents asynchronously from the given batch of urls
# Output will be list of Dictionary
async def extract_batch(urls: List[str], batch_num: int) -> List[dict[str, Any]]:
    """Extract documents from batch of documentation site URLs"""
    try:
        log_info(
            f"TavilyExtract: Processing batch {batch_num} with {len(urls)} URLs ",
            Colors.BLUE
        )
        # Await is for non-blocking operations
        docs = await tavily_extract.ainvoke(input={"urls": urls})
        log_success(f"TavilyExtract: Completed batch {batch_num} - extracted {len(docs.get('results',[]))} documents ")
        return docs
    except Exception as e:
        log_error(f"TavilyExtract: Failed to extract batch {batch_num} - {e}")
        return []

# Define a Coroutine Async Extract function
async def async_extract(url_batches: List[List[str]]):
    log_header("DOCUMENT EXTRACTION PHASE")
    log_info(
        f"TavilyExtract: Starting concurrent extraction of {len(url_batches)} batches",
        Colors.DARKCYAN
    )
    # A coroutine is a function that can pause and resume without blocking the rest of your program.
    # A lightweight thread managed by Python, not the OS.
    # Coroutines let Python do this: start → wait for I/O → pause → let other tasks run → resume → finish

    # First Line: Coroutine will be stored in a variable called tasks, Loops through each batch of URLs,
    # Calls the async function extract_batch(...) along with batch id, But does NOT run it yet Instead,
    # it creates a list of coroutine objects, for e.g., Batch 1 → extract_batch(batch1, 1),
    # Batch 2 → extract_batch(batch2, 2), Batch 3 → extract_batch(batch3, 3)
    # Second line: running all tasks concurrently, Start and execute them at the same time
    # Waits for all of batches to finish, Returns their results in a list
    tasks = [extract_batch(batch, i+1) for i, batch in enumerate(url_batches)]
    # Here *tasks equals to calling asyncio.gather(task1, task2, task3),
    # return_exceptions=True -> in case of error, return it as results, and let other batches continue
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions & successful pages (List of linkchain Document which is a dictionary
    # with content and metadata (url source) )
    all_pages = []
    failed_batches = 0
    for result in results:
        if isinstance(result, Exception):
            log_error(f"TavilyExtract: Batch failed with exception {result} ")
            failed_batches+= 1
        else:
            for extracted_page in result['results']:
                document = Document(
                    page_content=extracted_page['raw_content'],
                    metadata={"source": extracted_page['url']},
                )
                all_pages.append(document)
    log_success(f"TavilyExtract: Extraction Completed! total pages extracted: {len(all_pages)} batches")
    if failed_batches > 0:
        log_warning(f"TavilyExtract: {failed_batches} batches failed during the extraction")
    return all_pages

# Helper coroutine function to index the all linkchain documents asynchronously
async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batches asynchronously"""
    log_header("VECTOR STORAGE PHASE")
    log_info(f"VectorStore Indexing: Preparing to add {len(documents)} documents into Vector store.")

    # Create batches
    batches = [
        documents[i: i+batch_size ] for i in range(0, len(documents), batch_size)
    ]

    log_info(f"Vectorstore Indexing: Split into {len(batches)} of batch size {batch_size} documents each")

    # Another Helper coroutine function inside above function to process the batches concurrently
    # This new coroutine is visible only to index_documents_async function, not outside
    # This function receives list of documents and process each one of them
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorStore.aadd_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} "
                f"({len(batch)} documents)"
            )
        except Exception as e:
            log_error(f"Vectorstore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    # Now process the batches concurrently, first create variable of coroutine of all batch tasks
    # and then execute one by one, using gather, to fire up all the coroutines concurrently
    # The results variable is going to have list of booleans
    tasks = [add_batch(batch, i+1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    succesful = sum(1 for result in results if result is True)
    if succesful == len(batches):
        log_success(f"Vectorstore Indexing: All batches processed successfully {succesful/len(batches)}")
    else:
        log_warning(f"Vectorstore Indexing: Processed {succesful/len(batches)} batches successfully")

# Crawling LangChain Documentation
# Web crawling refers to the automated process of browsing website by following hyperlinks, clicking them,
# going to from page to another page and uncovering more and more related contents. For Agents and autonomous
# agents, crawling is a key capability, especially accessing deeper layers of the web which not reachable
# through standard search.

async def main():
    """Main async function to orchestrate the entire process."""
    #print(f"PineCone Key {os.environ['PINECONE_API_KEY']}")
    log_header("DOCUMENTATION INGESTION PIPELINE")
    # **************************************************************************************
    # USING TailyCrawl
    # **************************************************************************************
    #log_info("TailyCrawl: Starting to crawl documentation from https://python.langchain.com/", Colors.PURPLE)

    # Use Tavily craw object, a link chain tool, so invoke it with the url to crawl.  This tool crawls hundreds of paths
    # in parallel with built in extraction and intelligent discovery
    # res = tavily_crawl.invoke({
    #     "url": "https://python.langchain.com/",
    #     "max_depth": 1, # To set how far from base url, the crawl can explore (default is 1, can set 5), bigger # will take longer time to crawl
    #     "extract_depth": "advanced", # Advanced mode will help to retrieve more data including tables, embedded content with higher success rate, but may increase latency
    #     "instructions": "content on ai agents" # This instructions will help Tavily whether to crawl that page or not based the given content related to AI agents
    # })
    # That line converts each Tavily search result into a LangChain Document object, storing the page’s raw HTML/text as page_content and a source
    # dictionary with URL as metadata to highlight where the source for RAG comes from, this creates trust in our system
    # all_docs = [
    #     Document(
    #         page_content=result['raw_content'],
    #         metadata={"source": result['url']}
    #     )
    #     for result in res["results"]]
    # **************************************************************************************
    # Using Tavily Map
    # **************************************************************************************
    log_info(
        "TavilyMap: Starting to map documentation structure from https://python.langchain.com/",
        Colors.PURPLE,
    )
    site_map = tavily_map.invoke("https://python.langchain.com/")
    log_success(
        f"TavilyMap: Successfully mapped {len(site_map['results'])} URLs from documentation site"
    )

    # Split URLs into batches of 10
    url_batches = chunk_urls(list(site_map["results"]), chunk_size=10)
    log_info(
        f"URL Processing: Split {len(site_map['results'])} URLs into {len(url_batches)} batches",
        Colors.BLUE,
    )
    # Extract documents from URLs
    all_docs = await async_extract(url_batches)
    # **************************************************************************************

    # Split documents into chunks recursively based on semantic/structural sentences and
    # paragraphs and convert them into linkchain documents
    log_success("DOCUMENT CHUNKING PHASE")
    log_info(f"Text Splitter: processing {len(all_docs)} documents with 4000 (chars) chunk size and 200 (chars) overlap",
             Colors.YELLOW)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    # split_documents returns list of linkchain documents
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(f"Text Splitter: Created {len(splitted_docs)} documents from {len(all_docs)} documents")

    # Now Turn these splitted docs into vector and store them into vector store asynchronously
    # Take care of the batch_size(100), too big will results into rate_limit checks
    await index_documents_async(splitted_docs, batch_size=100)

    log_header("PINECONE COMPLETE")
    log_success("Documentation Ingestion is completed successfully!")
    log_info("Summary: ", Colors.BOLD)
    log_info(f"   * URLs mapped: {len(site_map['results'])}")
    log_info(f"   * Documents extracted: {len(all_docs)}")
    log_info(f"   * Chunks created: {len(splitted_docs)}")

if __name__ == "__main__":

    asyncio.run(main())