# This file is optional, deviation from the main application build process
# This uses Tavily map and extract, to transform scraping results into langchain document
# using coroutine, concurrent batche processing.  This a good example to go through python strengths

# ==========================================================================================
# Below section is option to demonstrate Tavily Map and Tavily Extract
# ==========================================================================================
# Tavily Map is going to discover and map out the linkchain documentation and website, urls, that we want to scrape and extract info from
# Tavily Extract is going to extract the data of that pages that holds all the information
# Test out this using Google's colab runtime, Google Colab runtimes provide the dedicated cloud-based
# virtual environments where your notebook code executes. https://colab.research.google.com/
# ***** Refer to tavily.py for more details *****
# ==========================================================================================


import asyncio
from typing import List, Any

from langchain_core.documents import Document

# Import all logging functions from logger.py
from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)
from langchain_tavily import TavilyExtract, TavilyMap, tavily_extract
import ssl
import certifi
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure SSL context with a valid certificate using certifi package. This is for making tons of Tavily API requests
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# -------------------------------------------------
# 1. DEFINE ALL HELPER FUNCTIONS BEFORE USING THEM
# -------------------------------------------------
def chunk_urls(urls: List[str], chunk_size: int=10) -> List[List[str]]:
    """Split the list of of URLs into chunks of specified size"""
    chunks = []
    for i in range (0, len(urls), chunk_size):
        chunks.append(urls[i:i+chunk_size])
    return chunks

async def main():
    """ Main async function to orchestrate entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info("TavilyMap: Starting to map documentation structure from https://python.langchain.com",
             Colors.PURPLE)
    # Extract documents from URLs
    all_docs = await async_extract(url_batches)
    print("Processing Completed")

# Below Coroutine Concurrent Extract batch code receives list of urls and batch number (for observability)
# Extract the document contents asynchronously from the given batch of urls
# Output will be list of Dictionary
async def extract_batch(urls: List[str], batch_num: int) -> List[dict[str, Any]]:
    """Extract documents from batch of documentation site URLs"""
    try:
        log_info(
            f"TavilyExtract: Processing batch {batch_num} with {len(urls)} URLs ",
            Colors.BLUE
        )
        # Await is for non blocking operations
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
    # return_exceptions=True -> in case of error, return it as results, and other batches continue
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

# -----------------------------------------
# 2. MAIN EXECUTION STARTS HERE
# -----------------------------------------

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
print("Initialisation of object activity is completed")

# Get the site map with all the urls
site_map = tavily_map.invoke("https://python.langchain.com")
log_success(f"TavilyMap: Successfully mapped {len(site_map['results'])} URLs from the documentation site")

# Split the urls into batches of 10
url_batches = chunk_urls(list(site_map["results"]), chunk_size=10)
log_info(f"URL Processing: Split {len(site_map['results'])} URLs into {len(url_batches)} batches",
         Colors.BLUE)

if __name__ == "__main__":
    asyncio.run(main())

