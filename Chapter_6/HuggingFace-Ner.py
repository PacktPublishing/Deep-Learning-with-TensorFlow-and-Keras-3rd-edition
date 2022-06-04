from transformers import pipeline
ner_pipe = pipeline("ner")
sequence = """Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much."""
for entity in ner_pipe(sequence):
    print(entity) 
