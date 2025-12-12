

import create_vectordb_single, create_vectordb_flow, create_vectordb_hybrid
import sys
from utils import chdir_in_project

chdir_in_project("vectorstore/vectorstore_generators")

print("\n===================================")
print("Running create_vectordb_single...")
print("===================================")
create_vectordb_single.main()

print("\n===================================")
print("Running create_vectordb_flow...")
print("===================================")

original_argv = sys.argv
for num_utterances in [3, 5, 7, 10, 12]:
    sys.argv = ['create_vectorstore.py', '--num_utterances', str(num_utterances)]
    # Run the main function, which will now parse the new arguments
    print("\n===================================")
    print(f"Running create_vectordb_flow with --num_utterances: {num_utterances}...")
    print("===================================")
    create_vectordb_flow.main()
# Restore original arguments
sys.argv = original_argv


print("\nRunning create_vectordb_hybrid...")
original_argv = sys.argv
for num_utterances in [3, 5, 7, 10, 12]:
    sys.argv = ['create_vectorstore.py', '--num_utterances', str(num_utterances)]
    # Run the main function, which will now parse the new arguments
    print("\n===================================")
    print(f"Running create_vectordb_hybrid with --num_utterances: {num_utterances}...")
    print("===================================")
    create_vectordb_hybrid.main()
# Restore original arguments
sys.argv = original_argv

print("\n--- Runner script has finished. ---")


