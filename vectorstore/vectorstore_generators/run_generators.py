

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










#
# # new_runner.py
#
# # No need to manipulate sys or os/chdir
# import create_vectordb_single
# import create_vectordb_flow
# import create_vectordb_hybrid
#
# print("--- Starting Vector DB Creation ---")
#
# # --- 1. Run the single utterance script ---
# print("\nRunning create_vectordb_single...")
# try:
#     create_vectordb_single.main()
#     print("✅ Success.")
# except Exception as e:
#     print(f"❌ FAILED: {e}")
#
#
# # --- 2. Run the flow script with multiple window sizes ---
# print("\nRunning create_vectordb_flow for multiple window sizes...")
# for num_utterances in [3, 5, 7, 10, 12]:
#     print(f"  - Running with --num_utterances: {num_utterances}...")
#     try:
#         # Pass arguments as a list of strings
#         cli_args = ['--num_utterances', str(num_utterances)]
#         create_vectordb_flow.main(cli_args=cli_args)
#         print(f"  ✅ Success for {num_utterances} utterances per page content.")
#     except Exception as e:
#         print(f"  ❌ FAILED for {num_utterances} utterances per page content: {e}")
#
#
# # --- 3. Run the hybrid script with multiple window sizes ---
# print("\nRunning create_vectordb_hybrid for multiple window sizes...")
# for num_utterances in [3, 5, 7, 10, 12]:
#     print(f"  - Running with --num_utterances: {num_utterances}...")
#     try:
#         cli_args = ['--num_utterances', str(num_utterances)]
#         create_vectordb_hybrid.main(cli_args=cli_args)
#         print(f"  ✅ Success for {num_utterances} utterances per page content")
#     except Exception as e:
#         print(f"  ❌ FAILED for {num_utterances} utterances per page content: {e}")
#
#
# print("\n--- Runner script has finished. ---")