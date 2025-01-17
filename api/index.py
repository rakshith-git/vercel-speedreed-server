from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
import os
import logging
from logging.handlers import RotatingFileHandler
from werkzeug.middleware.proxy_fix import ProxyFix

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Add file handler for logging
file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
logger.addHandler(file_handler)

# Initialize Flask app
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)  # For proper handling behind proxy servers

# Configure CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# Load environment variables or use defaults
PORT = int(os.environ.get('PORT', 5000))
DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
HOST = os.environ.get('HOST', '0.0.0.0')

try:
    # Load spaCy model
    logger.info("Loading spaCy model...")
    nlp = spacy.load('en_core_web_md')
    logger.info("spaCy model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {str(e)}")
    raise

# Define tag groups for RSVP
DELAY_GROUPS = {
    'named_entity': [],
    'content': ['NOUN', 'PROPN', 'VERB', 'ADJ', 'INTJ'],
    'function': ['DET', 'PRON', 'ADP', 'AUX', 'CCONJ', 'SCONJ'],
    'modifier': ['ADV', 'NUM'],
    'punctuation': ['PUNCT', 'SPACE', 'SYM']
}

def get_group(token):
    """Return the group name for a given token, considering both NER and POS"""
    if token.ent_type_:
        return 'named_entity'
    
    pos_tag = token.pos_
    for group, tags in DELAY_GROUPS.items():
        if pos_tag in tags:
            return group
    return 'other'

def combine_with_punctuation(doc):
    """Combine tokens with following punctuation and get their groups"""
    combined_tokens = []
    combined_groups = []
    current_token = ""
    current_group = None
    
    for i, token in enumerate(doc):
        if token.pos_ in DELAY_GROUPS['punctuation']:
            if current_token:
                current_token += token.text
            else:
                current_token = token.text
                current_group = get_group(token)
        else:
            if current_token:
                combined_tokens.append(current_token)
                combined_groups.append(current_group)
            current_token = token.text
            current_group = get_group(token)
            
        if i == len(doc) - 1:
            combined_tokens.append(current_token)
            combined_groups.append(current_group)
            
    return combined_tokens, combined_groups

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model': 'en_core_web_md'})

@app.route('/pos-tag', methods=['POST'])
def pos_tag():
    try:
        data = request.get_json()
        
        if not data:
            logger.warning("No JSON data in request")
            return jsonify({'error': 'No JSON data provided'}), 400
            
        if 'text' not in data:
            logger.warning("No 'text' field in request data")
            return jsonify({'error': 'Please provide text in the request body'}), 400
            
        text = data['text']
        
        if not text.strip():
            logger.warning("Empty text provided")
            return jsonify({'error': 'Text cannot be empty'}), 400

        logger.info(f"Processing text of length: {len(text)}")
        doc = nlp(text)
        
        tokens, groups = combine_with_punctuation(doc)
        
        # entities_info = [
        #     {
        #         'text': ent.text,
        #         'label': ent.label_
        #     } for ent in doc.ents
        # ]
        
        response = {
            'tokens': tokens,
            'groups': groups,
            # 'entities_found': entities_info
        }
        
        logger.info(f"Successfully processed text with {len(tokens)} tokens")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error('Server Error', exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info(f"Starting server on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=DEBUG)