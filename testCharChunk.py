from langchain_text_splitters import RecursiveCharacterTextSplitter
from termcolor import colored

with open("long_text.txt", "r", encoding="utf-8") as f:
    long_text = f.read()

overlap = 70

def find_overlap(a, b):
    max_len = min(len(a), len(b))
    for i in range(max_len, 0, -1):
        if a.endswith(b[:i]):
            return b[:i]
    return ""

def main():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=overlap
    )
    chunks = text_splitter.split_text(long_text)
    # Option A: Print chunks with colored boundaries
    # for i, chunk in enumerate(chunks):
    #     print(colored(f"\n--- CHUNK {i+1} (len={len(chunk)}) ---", "yellow"))
    #     print(chunk)

    # Option B: To Highlight overlap regions
    for i, chunk in enumerate(chunks):
        print(f"\n--- CHUNK {i + 1} ---")

        if i > 0:
            prev = chunks[i - 1]
            overlap_text = find_overlap(prev, chunk)

            print(colored("OVERLAP:", "cyan"))
            print(colored(overlap_text, "cyan"))

        print(colored("CONTENT:", "green"))
        print(chunk)

    print("--- Completed ---")

if __name__ == "__main__":
    main()