from flask import Flask, request, jsonify, render_template
from calliope_logic import add_word_to_database, view_database, predict_blank_word_with_context
import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    """Serve the homepage."""
    return render_template('index.html')

@app.route('/calliope')
def calliope_page():
    """Serve the Calliope interface."""
    return render_template('calliope.html')

@app.route('/api/add_word', methods=['POST'])
def add_word():
    """API endpoint to add a word to the database."""
    data = request.json
    if not data or 'word' not in data:
        return jsonify({"status": "error", "message": "Missing 'word' in request."}), 400

    result = add_word_to_database(data['word'])
    return jsonify(result)

@app.route('/api/view_database', methods=['GET'])
def view_db():
    """API endpoint to view the database."""
    pos = request.args.get('pos')  # Optional POS filter
    if pos:
        result = view_database(pos)
        return jsonify({"status": "success", "data": result})
    else:
        grouped_data = view_database()
        return jsonify({"status": "success", "data": grouped_data})

@app.route('/api/predict_blank', methods=['POST'])
def predict_blank():
    """API endpoint to predict the word for a blank in a sentence."""
    data = request.json
    if not data or 'sentence' not in data:
        return jsonify({"status": "error", "message": "Missing 'sentence' in request."}), 400

    result = predict_blank_word_with_context(data['sentence'])
    return jsonify(result)

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if no PORT variable is set
    app.run(host='0.0.0.0', port=port)
