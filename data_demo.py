import pickle

# Initialize lists to store the three parts of data
mrna_sequences = []
species_ids = []
taxonomy = []
unknown_codon = []
# Read the data line by line
with open('../virus_rna_sequence_with_taxonomy.txt', 'r') as f:  # Replace with your actual file path
    for line in f:
        # Split the line into parts
        parts = line.strip().split(',')
        
        # First part is mRNA sequence
        mrna_sequences.append(parts[0])
        # print(len(parts[0]),len(parts[0])%3==0)
        if len(parts[0])%3!=0:
            unknown_codon.append(len(parts[0]))
        # Second part is Species ID
        species_ids.append(parts[1])
        
        # The rest is taxonomy information
        taxonomy.append(parts[2:])

# Save each list to a separate pickle file
with open('mrna_sequences.pkl', 'wb') as f:
    pickle.dump(mrna_sequences, f)

with open('species_ids.pkl', 'wb') as f:
    pickle.dump(species_ids, f)

with open('taxonomy.pkl', 'wb') as f:
    pickle.dump(taxonomy, f)

print("Data successfully split and saved to pickle files.")
print("Unknown codon length:", len(unknown_codon))