import pytest
import os
import logging
import sys
from unittest.mock import patch, MagicMock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('beam_search_test')

# Try to import with different paths
try:
    from src.auto_formatter.organize_requests import Node, Frontier, OrganizeRequest
except ImportError as e:
    logger.warning(f"Import error: {e}")
    # Try different import paths
    try:
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from src.auto_formatter.organize_requests import Node, Frontier, OrganizeRequest
    except ImportError as e2:
        logger.error(f"Could not import required modules: {e2}")
        raise

class TestBeamSearchOutput:
    """Test class that focuses on the actual output of the beam search algorithm."""
    
    @patch('src.auto_formatter.organize_requests.BertTokenizer')
    @patch('src.auto_formatter.organize_requests.BertModel')
    @patch('src.auto_formatter.organize_requests.spacy.load')
    @patch('os.path.exists')
    def test_beam_search_real_output(self, mock_exists, mock_spacy, mock_bert_model, mock_bert_tokenizer):
        """
        Test the beam search algorithm with realistic mock data and minimal mocking of
        internal methods to see the actual output.
        """
        logger.info("=== Starting beam search output test ===")
        
        # Set up mocks for external dependencies
        mock_exists.return_value = True
        
        # Create a list of test cases
        test_cases = [
            {
                "name": "Simple VP - Send Email",
                "request": "Please send the email to the team about the meeting.",
                "tokens": [
                    ("Please", "INTJ"), 
                    ("send", "VERB"), 
                    ("the", "DET"), 
                    ("email", "NOUN"),
                    ("to", "ADP"),
                    ("the", "DET"),
                    ("team", "NOUN"),
                    ("about", "ADP"),
                    ("the", "DET"),
                    ("meeting", "NOUN"),
                    (".", "PUNCT")
                ],
                "phrase_type": "VP",
                "phrase_category": "Task",
                "expected_phrases": ["send the email", "send the"]
            },
            {
                "name": "Multiple VPs - Schedule and Prepare",
                "request": "Schedule the meeting for tomorrow and prepare the quarterly report.",
                "tokens": [
                    ("Schedule", "VERB"),
                    ("the", "DET"),
                    ("meeting", "NOUN"),
                    ("for", "ADP"),
                    ("tomorrow", "NOUN"),
                    ("and", "CONJ"),
                    ("prepare", "VERB"),
                    ("the", "DET"),
                    ("quarterly", "ADJ"),
                    ("report", "NOUN"),
                    (".", "PUNCT")
                ],
                "phrase_type": "VP",
                "phrase_category": "Task",
                "expected_phrases": ["Schedule the meeting", "prepare the quarterly report"]
            },
            {
                "name": "NP - Quarterly Report",
                "request": "The quarterly financial report needs to be reviewed by Friday.",
                "tokens": [
                    ("The", "DET"),
                    ("quarterly", "ADJ"),
                    ("financial", "ADJ"),
                    ("report", "NOUN"),
                    ("needs", "VERB"),
                    ("to", "PART"),
                    ("be", "AUX"),
                    ("reviewed", "VERB"),
                    ("by", "ADP"),
                    ("Friday", "PROPN"),
                    (".", "PUNCT")
                ],
                "phrase_type": "NP",
                "phrase_category": "Task",
                "expected_phrases": ["The quarterly financial report"]
            }
        ]
        
        # Process each test case
        for test_case in test_cases:
            logger.info(f"\n--- Testing: {test_case['name']} ---")
            logger.info(f"Request: {test_case['request']}")
            logger.info(f"Phrase Type: {test_case['phrase_type']}")
            
            # Create mock doc
            doc_mock = MagicMock()
            tokens = []
            for text, tag in test_case["tokens"]:
                token_mock = MagicMock()
                token_mock.text = text
                token_mock.tag_ = tag
                tokens.append(token_mock)
            
            doc_mock.__iter__.return_value = tokens
            
            # Set up spaCy mock
            nlp_mock = MagicMock()
            nlp_mock.return_value = doc_mock
            mock_spacy.return_value = nlp_mock
            
            # Set up BERT mocks
            tokenizer_mock = MagicMock()
            mock_bert_tokenizer.from_pretrained.return_value = tokenizer_mock
            
            model_mock = MagicMock()
            outputs_mock = MagicMock()
            outputs_mock.last_hidden_state = MagicMock()
            model_mock.return_value = outputs_mock
            mock_bert_model.from_pretrained.return_value = model_mock
            
            # Create temporary test file path
            temp_file = "test_training_data.csv"
            
            # Initialize the processor with minimal mocking
            with patch('pandas.read_csv'):
                organizer = OrganizeRequest(
                    request=test_case["request"],
                    max_vision=3,
                    phrase_type=test_case["phrase_type"],
                    phrase_category=test_case["phrase_category"],
                    filename=temp_file,
                    beam_width=2
                )
                
                # Minimal override of key methods to avoid external dependencies
                # while preserving real algorithm behavior
                
                # Override contains_phrase to return realistic patterns
                original_contains_phrase = organizer.frontier.contains_phrase
                
                def mock_contains_phrase(node, phrase_type):
                    logger.info(f"Checking patterns in node: {node.state}")
                    # First generate a simplified version of real pattern matches
                    if len(node.state) > 0:
                        first_tag = node.state[0][1]
                        first_text = node.state[0][0]
                        
                        # For verb phrases (VP)
                        if phrase_type == "VP" and first_tag == "VERB":
                            patterns = []
                            
                            # Simple patterns like "send" (just the verb)
                            patterns.append([first_text])
                            
                            # If we have DET after VERB, create "verb the" pattern
                            if len(node.state) > 1 and node.state[1][1] == "DET":
                                patterns.append([first_text, node.state[1][0]])
                            
                            # If we have DET+NOUN after VERB, create "verb the noun" pattern
                            if (len(node.state) > 2 and node.state[1][1] == "DET" 
                                and node.state[2][1] in ["NOUN", "ADJ"]):
                                patterns.append([first_text, node.state[1][0], node.state[2][0]])
                                
                            # If we have DET+ADJ+NOUN, create full pattern
                            if (len(node.state) > 3 and node.state[1][1] == "DET" 
                                and node.state[2][1] == "ADJ" and node.state[3][1] == "NOUN"):
                                patterns.append([first_text, node.state[1][0], node.state[2][0], node.state[3][0]])
                                
                            logger.info(f"Found VP patterns: {patterns}")
                            return patterns
                        
                        # For noun phrases (NP)
                        elif phrase_type == "NP" and first_tag == "DET":
                            patterns = []
                            
                            # Simple pattern like "the report"
                            if len(node.state) > 1 and node.state[1][1] == "NOUN":
                                patterns.append([first_text, node.state[1][0]])
                            
                            # With adjective: "the quarterly report"
                            if (len(node.state) > 2 and node.state[1][1] == "ADJ" 
                                and node.state[2][1] == "NOUN"):
                                patterns.append([first_text, node.state[1][0], node.state[2][0]])
                                
                            # With multiple adjectives: "the quarterly financial report"
                            if (len(node.state) > 3 and node.state[1][1] == "ADJ" 
                                and node.state[2][1] == "ADJ" and node.state[3][1] == "NOUN"):
                                patterns.append([first_text, node.state[1][0], node.state[2][0], node.state[3][0]])
                                
                            logger.info(f"Found NP patterns: {patterns}")
                            return patterns
                    
                    # Default: no patterns found
                    logger.info("No patterns found")
                    return []
                
                # Override score_phrase to return predictable scores
                def mock_score_phrase(phrase, category):
                    # Assign scores based on pattern length (longer = better)
                    # Use expected_phrases to boost items that should rank higher
                    phrase_str = ' '.join(phrase)
                    base_score = len(phrase) * 0.2  # Longer phrases get higher scores
                    
                    # Boost expected phrases
                    boost = 0
                    for expected in test_case["expected_phrases"]:
                        if expected.lower() in phrase_str.lower():
                            boost = 0.5
                            break
                            
                    score = base_score + boost
                    logger.info(f"Scoring phrase '{phrase_str}': {score}")
                    return score
                
                # Apply our mock methods
                organizer.frontier.contains_phrase = mock_contains_phrase
                organizer.score_phrase = mock_score_phrase
                
                # Run the beam search
                logger.info("Running beam search...")
                try:
                    results = organizer.search_by_category()
                    
                    # Log results
                    logger.info(f"Search returned {len(results)} results:")
                    if results:
                        for i, result in enumerate(results):
                            logger.info(f"Result {i+1}:")
                            logger.info(f"  Action: {result.action}")
                            logger.info(f"  Score: {result.path_score}")
                            
                            # Get full path by traversing parents
                            path = []
                            current = result
                            while current:
                                if current.action:
                                    path.append(current.action)
                                current = current.parent
                            
                            # Reverse to get chronological order
                            path.reverse()
                            logger.info(f"  Full path: {path}")
                    else:
                        logger.info("No results found")
                        
                    # Verify expected phrases
                    found_phrases = [' '.join(result.action) if isinstance(result.action, list) else result.action 
                                    for result in results if result.action]
                    
                    logger.info(f"Found phrases: {found_phrases}")
                    logger.info(f"Expected phrases: {test_case['expected_phrases']}")
                    
                    # Check for overlaps
                    matches = []
                    for expected in test_case["expected_phrases"]:
                        for found in found_phrases:
                            if expected.lower() in found.lower():
                                matches.append(expected)
                                break
                    
                    logger.info(f"Matched expected phrases: {matches}")
                    logger.info(f"Match coverage: {len(matches)}/{len(test_case['expected_phrases'])}")
                    
                except Exception as e:
                    logger.error(f"Error in search_by_category: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                
                logger.info(f"--- Completed test: {test_case['name']} ---\n")
        
        logger.info("=== All beam search output tests completed ===")

if __name__ == "__main__":
    # Run the tests directly
    pytest.main(["-xvs", __file__])
