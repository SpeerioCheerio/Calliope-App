import spacy
import nltk
from nltk.corpus import wordnet as wn
from transformers import RobertaForMaskedLM, RobertaTokenizer
import torch
import sqlite3

# Initialize SQLite database connection
conn = sqlite3.connect('calliope_words.db', check_same_thread=False)
cursor = conn.cursor()

# Load SpaCy model
nltk.download('wordnet')
nltk.download('omw-1.4')
nlp = spacy.load('en_core_web_sm')

# Load RoBERTa model
roberta_model = RobertaForMaskedLM.from_pretrained('roberta-base')
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def fetch_definitions(word):
    """Fetch up to two definitions for a word using WordNet."""
    synsets = wn.synsets(word)
    return [synset.definition() for synset in synsets[:2]] or ["Definition not found."]

def determine_pos(word):
    """Determine the part of speech (POS) of a word using SpaCy."""
    doc = nlp(word)
    pos_mapping = {"NOUN": "noun", "VERB": "verb", "ADJ": "adjective", "ADV": "adverb"}
    return pos_mapping.get(doc[0].pos_, "other")

def add_word_to_database(word):
    """Add a word to the database with its POS and definitions."""
    pos = determine_pos(word)
    definitions = fetch_definitions(word)

    try:
        cursor.execute('''
            INSERT INTO words (word, pos, definition1, definition2)
            VALUES (?, ?, ?, ?)
        ''', (word, pos, definitions[0], definitions[1] if len(definitions) > 1 else "Definition not found."))
        conn.commit()
        return {"status": "success", "message": f"Word '{word}' added to the database under POS '{pos}'."}
    except sqlite3.IntegrityError:
        return {"status": "error", "message": f"Word '{word}' already exists in the database."}

def view_database(pos=None):
    """View words in the database, filtered by POS if specified."""
    query = "SELECT word, pos, definition1, definition2 FROM words"
    params = ()
    if pos and pos in ["noun", "verb", "adjective", "adverb", "other"]:
        query += " WHERE pos = ?"
        params = (pos,)
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    if pos:
        return [{"word": row[0], "pos": row[1], "definitions": [row[2], row[3]]} for row in rows]
    else:
        # Group words by POS
        grouped_data = {"noun": [], "verb": [], "adjective": [], "adverb": [], "other": []}
        for row in rows:
            grouped_data[row[1]].append({"word": row[0], "definitions": [row[2], row[3]]})
        return grouped_data


def predict_blank_word_with_context(sentence):
    """Predict the word for the blank ('[BLANK]') in a sentence."""
    if "[BLANK]" not in sentence:
        return {"status": "error", "message": "Invalid format. Use '[BLANK]' to indicate the blank."}
    
    masked_sentence = sentence.replace("[BLANK]", "<mask>")
    inputs = roberta_tokenizer(masked_sentence, return_tensors="pt")
    mask_token_index = torch.where(inputs.input_ids == roberta_tokenizer.mask_token_id)[1]

    with torch.no_grad():
        logits = roberta_model(**inputs).logits

    mask_logits = logits[0, mask_token_index, :].squeeze(0)
    top_tokens = torch.topk(mask_logits, 100)  # Increase the pool for more robust predictions
    token_indices = top_tokens.indices.tolist()

    cursor.execute('SELECT word FROM words')
    database_words = {row[0] for row in cursor.fetchall()}

    predictions = [
        (roberta_tokenizer.decode([token]).strip(), torch.softmax(mask_logits, dim=-1)[token].item())
        for token in token_indices
    ]
    filtered_predictions = [
        {"word": word, "probability": prob}
        for word, prob in predictions if word in database_words
    ][:5]  # Ensure at least 5 predictions

    # Fill with additional words if fewer than 5 are found
    if len(filtered_predictions) < 5:
        additional_words = [
            {"word": word, "probability": 0.0}
            for word in database_words
            if word not in [pred["word"] for pred in filtered_predictions]
        ]
        filtered_predictions.extend(additional_words[:5 - len(filtered_predictions)])

    return {"status": "success", "predictions": filtered_predictions}


def close_connection():
    """Close the database connection."""
    conn.close()

# Ensure proper database closure on script exit
import atexit
atexit.register(close_connection)
