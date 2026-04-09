import os
import re
import json

input_folder = "data/cleaned"
output_folder = "data/chunks"
os.makedirs(output_folder, exist_ok=True)


# Pattern to detect regulatory section numbers
SECTION_PATTERN = re.compile(r'^(\d+(\.\d+)*\.?)\s+')

# Pattern to detect all caps headings (common in RBI docs)
HEADING_PATTERN = re.compile(r'^[A-Z][A-Z\s\-]{5,}$')


def split_into_structured_blocks(text):
    lines = text.split("\n")
    blocks = []
    current_block = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # If new section or heading detected → start new block
        if SECTION_PATTERN.match(line) or HEADING_PATTERN.match(line):
            if current_block:
                blocks.append("\n".join(current_block))
                current_block = []

        current_block.append(line)

    if current_block:
        blocks.append("\n".join(current_block))

    return blocks


def merge_blocks_to_token_limit(blocks, max_words=500, overlap_words=75):
    merged_chunks = []
    current_chunk = []
    word_count = 0

    for block in blocks:
        block_words = len(block.split())

        if word_count + block_words <= max_words:
            current_chunk.append(block)
            word_count += block_words
        else:
            chunk_text = "\n\n".join(current_chunk)
            merged_chunks.append(chunk_text)

            # create overlap from end
            words = chunk_text.split()
            overlap = words[-overlap_words:] if len(words) > overlap_words else words
            current_chunk = [" ".join(overlap), block]
            word_count = len(overlap) + block_words

    if current_chunk:
        merged_chunks.append("\n\n".join(current_chunk))

    return merged_chunks


# Process each cleaned file
for file in os.listdir(input_folder):
    if file.endswith(".txt"):
        path = os.path.join(input_folder, file)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        structured_blocks = split_into_structured_blocks(text)
        final_chunks = merge_blocks_to_token_limit(structured_blocks)

        # Save chunks as JSON
        output_path = os.path.join(output_folder, file.replace(".txt", "_chunks.json"))

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_chunks, f, indent=2, ensure_ascii=False)

        print(f"Chunked: {file} | Total Chunks: {len(final_chunks)}")
