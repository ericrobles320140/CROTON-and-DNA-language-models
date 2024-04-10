#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 15:51:21 2024

@author: omarmahmood
"""

import numpy as np
import random
from Bio import SeqIO

def split_dataset(sequences, train_frac=0.7, valid_frac=0.15):

    # Shuffle the list of sequences to ensure random distribution
    random.shuffle(sequences)
    
    # Calculate split indices
    total_sequences = len(sequences)
    train_end = int(total_sequences * train_frac)
    valid_end = train_end + int(total_sequences * valid_frac)
    
    # Split the sequences
    train_sequences = sequences[:train_end]
    valid_sequences = sequences[train_end:valid_end]
    test_sequences = sequences[valid_end:]
    
    return train_sequences, valid_sequences, test_sequences

def one_hot_encode_sequence(sequence):
    nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    one_hot_sequence = np.zeros((len(sequence), 5))
    
    for i, nucleotide in enumerate(sequence):
        if nucleotide in nucleotide_map:
            one_hot_sequence[i, nucleotide_map[nucleotide]] = 1
    return one_hot_sequence

def find_sequence(sequences, query_sequence):
    found_sequences = {}
    for seq_record in sequences:
        sequence_str = str(seq_record.seq).upper()
        if query_sequence.upper() in sequence_str:
            start_pos = sequence_str.find(query_sequence.upper())
            found_sequences[seq_record.id] = start_pos
    return found_sequences

def pad_sequences(encoded_sequences, desired_length, padding_value=0):
    padded_sequences = np.full((len(encoded_sequences), desired_length, 5), padding_value)
    for i, sequence in enumerate(encoded_sequences):
        sequence_length = min(len(sequence), desired_length)
        padded_sequences[i, :sequence_length, :] = sequence[:sequence_length]
    return padded_sequences

def print_encoding_key():
    print("One-hot encoding key:")
    print("[A, C, G, T, N]")  # Updated to include N
    print("[1, 0, 0, 0, 0] = A")
    print("[0, 1, 0, 0, 0] = C")
    print("[0, 0, 1, 0, 0] = G")
    print("[0, 0, 0, 1, 0] = T")
    print("[0, 0, 0, 0, 1] = N\n")  # Explanation for N

def main():
    fasta_file_path = input("Enter the path to your FASTA file: ")
    all_sequences = list(SeqIO.parse(fasta_file_path, "fasta"))
    
    # Split the dataset
    train_sequences, valid_sequences, test_sequences = split_dataset(all_sequences)
    
    print(f"Total sequences: {len(all_sequences)}")
    print(f"Training sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(valid_sequences)}")
    print(f"Test sequences: {len(test_sequences)}")

    for seq in train_sequences[:5]:
        print(seq.id)
    print_encoding_key()  # Print the encoding key at the beginning
    
    choice = input("Do you want to search for a specific sequence? (yes/no): ")
    fasta_file_path = input("Enter the path to your FASTA file: ")
    sequences = list(SeqIO.parse(fasta_file_path, "fasta"))
    
    if choice.lower() == 'yes':
        query_sequence = input("Enter the sequence you're looking for: ")
        found_sequences = find_sequence(sequences, query_sequence)
        
        if found_sequences:
            for seq_id, start_pos in found_sequences.items():
                print(f"Found sequence in record {seq_id} starting from position {start_pos + 1}")
                encoded_seq = one_hot_encode_sequence(query_sequence)
                print("One-hot encoding of the found sequence:")
                for i, encoded_nucleotide in enumerate(encoded_seq):
                    print(f"Position {start_pos + i + 1}: {encoded_nucleotide}")
        else:
            print("The sequence was not found in any record.")
    else:
        desired_length = int(input("Enter the desired length for encoding: "))
        for seq_record in sequences:
            encoded_sequence = one_hot_encode_sequence(str(seq_record.seq))
            padded_sequence = pad_sequences([encoded_sequence], desired_length)[0]
            print(f"Encoded and padded sequence for record {seq_record.id}:")
            for i in range(min(desired_length, len(encoded_sequence))):
                print(encoded_sequence[i])
            print(f"First padded sequence shape: {padded_sequence.shape}")

if __name__ == "__main__":
    main()
