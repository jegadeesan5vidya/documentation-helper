from sentence_transformers import SentenceTransformer, util
import nltk
import difflib

# nltk.download('punkt')
# nltk.download('punkt_tab')
# Alternative is to run command in terminal: python -m nltk.downloader punkt punkt_tab
with open("long_text.txt", "r", encoding="utf-8") as f:
    long_text = f.read()

model = SentenceTransformer("all-MiniLM-L6-v2")  # fast 384-dim model

def semantic_chunk(text, threshold=0.6, max_chunk_size=2000):
    sentences = nltk.sent_tokenize(text)
    embeddings = model.encode(sentences, convert_to_tensor=True)

    chunks = []
    current_chunk = sentences[0]

    for i in range(1, len(sentences)):
        sim = util.cos_sim(embeddings[i], embeddings[i-1]).item()

        # If similarity drops → new chunk
        if sim < threshold or len(current_chunk) > max_chunk_size:
            chunks.append(current_chunk)
            current_chunk = sentences[i]
        else:
            current_chunk += " " + sentences[i]

    chunks.append(current_chunk)
    return chunks

def diff_text(original, chunk):
    diff = difflib.unified_diff(
        original.splitlines(),
        chunk.splitlines(),
        lineterm=""
    )
    return "\n".join(diff)



if __name__ == "__main__":
    chunks = semantic_chunk(long_text)
    # Option 0: Standard display of resulted chunks
    # for i, c in enumerate(chunks):
    #     print(f"\n--- SEMANTIC CHUNK {i+1} ---\n{c}\n")

    # Option 1 — Show a diff between original text and each semantic chunk, a Git‑style diff
    # for i, chunk in enumerate(chunks):
    #     print(f"\n=== DIFF FOR SEMANTIC CHUNK {i + 1} ===")
    #     print(diff_text(long_text, chunk))

    # Option 2 — Highlight where each semantic chunk appears in the original
    # If idx == -1, the chunk was reconstructed (common with semantic grouping).
    # for i, chunk in enumerate(chunks):
    #     idx = long_text.find(chunk[:50])  # search first 50 chars
    #     print(f"\nChunk {i + 1} starts at index: {idx}")

    # Option 3 — Visualize semantic chunk boundaries inside the original text
    # This overlays chunk markers directly into the original text.
    marked = long_text
    for i, chunk in enumerate(chunks):
        snippet = chunk[:40]  # first 40 chars
        marked = marked.replace(snippet, f"\n\n<<< CHUNK {i + 1} >>>\n{snippet}")

    print(marked)


    print("--- Completed ---")