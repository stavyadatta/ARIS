from fuzzywuzzy import fuzz

name1 = "pamela"
name2 = "sonal"

# Approximate string similarity
similarity = fuzz.ratio(name1, name2)

print("Similarity:", similarity)  # Typically >90%
