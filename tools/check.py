import joblib
import os

# Check if the files exist
words_file_path = "../tools/word_data.pkl"
authors_file_path = "../tools/email_authors.pkl"

print("Checking if files exist...")

if not os.path.exists(words_file_path):
    print(f"File not found: {words_file_path}")
else:
    print(f"Found file: {words_file_path}")

if not os.path.exists(authors_file_path):
    print(f"File not found: {authors_file_path}")
else:
    print(f"Found file: {authors_file_path}")

print("Attempting to load word data...")

try:
    # Load the word data
    with open(words_file_path, "rb") as words_file_handler:
        word_data = joblib.load(words_file_handler)
    print("Loaded word data successfully.")
except Exception as e:
    print(f"Error loading word data: {e}")

print("Attempting to load authors data...")

try:
    # Load the authors data
    with open(authors_file_path, "rb") as authors_file_handler:
        authors = joblib.load(authors_file_handler)
    print("Loaded authors data successfully.")
except Exception as e:
    print(f"Error loading authors data: {e}")

# Print some information about the data if loaded successfully
if 'word_data' in locals() and 'authors' in locals():
    print(f"Number of emails: {len(word_data)}")
    print(f"Number of labels: {len(authors)}")
    if len(word_data) > 0 and len(authors) > 0:
        print(f"Sample email text: {word_data[0]}")
        print(f"Sample label: {authors[0]}")
else:
    print("Data not loaded properly.")