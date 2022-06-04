from transformers import pipeline
generator = pipeline(task="text-generation")
generator("Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone")
