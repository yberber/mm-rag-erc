
import sys

from src.vectorstore.generators import build_single_vectorstore, build_flow_vectorstore, build_hybrid_vectorstore


print("\n===================================")
print("Running build_single_vectorstore...")
print("===================================")
build_single_vectorstore.main()

print("\n===================================")
print("Running build_flow_vectorstore...")
print("===================================")

original_argv = sys.argv
for num_utterances in [3, 5, 7, 10, 12]:
    sys.argv = ['create_vectorstore.py', '--num_utterances', str(num_utterances)]
    # Run the main function, which will now parse the new arguments
    print("\n===================================")
    print(f"Running build_flow_vectorstore with --num_utterances: {num_utterances}...")
    print("===================================")
    build_flow_vectorstore.main()
# Restore original arguments
sys.argv = original_argv


print("\nRunning build_hybrid_vectorstore...")
original_argv = sys.argv
for num_utterances in [3, 5, 7, 10, 12]:
    sys.argv = ['create_vectorstore.py', '--num_utterances', str(num_utterances)]
    # Run the main function, which will now parse the new arguments
    print("\n===================================")
    print(f"Running build_hybrid_vectorstore with --num_utterances: {num_utterances}...")
    print("===================================")
    build_hybrid_vectorstore.main()
# Restore original arguments
sys.argv = original_argv

print("\n--- Runner script has finished. ---")


