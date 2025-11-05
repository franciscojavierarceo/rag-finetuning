#!/usr/bin/env python3
"""
Script to chunk parquet data for Feast/Milvus compatibility.
This script reads a parquet file and chunks the text field to ensure it doesn't exceed
the Milvus varchar field maximum length (512 characters).
"""

import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import os

def chunk_text(text, max_chars=380):
    """
    Chunk text to ensure it doesn't exceed max_chars while preserving whole words.

    Args:
        text (str): The text to chunk
        max_chars (int): Maximum characters per chunk

    Returns:
        list: List of text chunks
    """
    if not text or len(text) <= max_chars:
        return [text]

    words = text.split()
    if not words:
        return [text]

    chunks = []
    current_chunk_words = []

    for word in words:
        # Check if adding the next word exceeds the character limit
        potential_chunk = ' '.join(current_chunk_words + [word])
        if len(potential_chunk) > max_chars:
            # If the current chunk is valid, save it
            if current_chunk_words:
                chunk_text = ' '.join(current_chunk_words)
                chunks.append(chunk_text)
            # Start a new chunk with the current word
            current_chunk_words = [word]
        else:
            current_chunk_words.append(word)

    # Add the last remaining chunk
    if current_chunk_words:
        chunk_text = ' '.join(current_chunk_words)
        chunks.append(chunk_text)

    return chunks

def chunk_parquet_data(input_file, output_file, max_chars=380):
    """
    Read parquet file, chunk the text field, and save to new parquet file.

    Args:
        input_file (str): Path to input parquet file
        output_file (str): Path to output parquet file
        max_chars (int): Maximum characters per text chunk
    """
    print(f"Reading parquet file: {input_file}")
    df = pd.read_parquet(input_file)

    print(f"Original data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Check if text column exists
    if 'text' not in df.columns:
        raise ValueError("'text' column not found in the parquet file")

    # Check max text length in original data
    max_text_length = df['text'].str.len().max()
    print(f"Maximum text length in original data: {max_text_length}")

    # Count how many rows exceed the limit
    exceeding_rows = (df['text'].str.len() > 512).sum()
    print(f"Number of rows exceeding 512 characters: {exceeding_rows}")

    chunked_data = []

    for idx, row in df.iterrows():
        text = row['text']
        chunks = chunk_text(text, max_chars)

        for chunk_idx, chunk in enumerate(chunks):
            new_row = row.copy()
            new_row['text'] = chunk
            # Create unique ID for each chunk
            new_row['id'] = f"{row['id']}_{chunk_idx + 1}" if len(chunks) > 1 else row['id']
            chunked_data.append(new_row)

    # Create new dataframe
    chunked_df = pd.DataFrame(chunked_data)

    print(f"Chunked data shape: {chunked_df.shape}")

    # Verify no text exceeds max_chars
    max_chunked_length = chunked_df['text'].str.len().max()
    print(f"Maximum text length after chunking: {max_chunked_length}")

    if max_chunked_length > max_chars:
        print(f"WARNING: Some chunks still exceed {max_chars} characters!")

    # Save to new parquet file
    print(f"Saving chunked data to: {output_file}")
    chunked_df.to_parquet(output_file, index=False)

    print("Chunking complete!")
    return chunked_df

def main():
    parser = argparse.ArgumentParser(description='Chunk parquet data for Feast/Milvus compatibility')
    parser.add_argument('--input', '-i', required=True, help='Input parquet file path')
    parser.add_argument('--output', '-o', help='Output parquet file path (default: input_chunked.parquet)')
    parser.add_argument('--max-chars', '-m', type=int, default=380,
                       help='Maximum characters per chunk (default: 380)')

    args = parser.parse_args()

    input_file = args.input
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist")
        return 1

    if args.output:
        output_file = args.output
    else:
        # Create output filename by adding _chunked before the extension
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_chunked{ext}"

    try:
        chunk_parquet_data(input_file, output_file, args.max_chars)
        print(f"\nYou can now update your feature_store.yaml to use: {output_file}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())