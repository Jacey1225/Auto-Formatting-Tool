import pytest

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.auto_formatter.organize_requests import OrganizeRequest, Node, Frontier
import spacy
import logging

@pytest.fixture
def setup_organize_request():
    """Fixture to set up an OrganizeRequest instance."""
    request = "John will complete the project documentation by Friday."
    max_vision = 5
    phrase_type = "NP"
    phrase_category = "Task"
    beam_width = 3
    return OrganizeRequest(request, max_vision, phrase_type, phrase_category, beam_width)

def test_node_initialization():
    """Test the initialization of the Node class."""
    node = Node(state=[("John", "PROPN"), ("will", "AUX")], parent=None, action=None, path_score=0)
    assert node.state == [("John", "PROPN"), ("will", "AUX")]
    assert node.parent is None
    assert node.action is None
    assert node.path_score == 0

def test_frontier_add_and_pop():
    """Test adding and popping nodes from the Frontier."""
    frontier = Frontier()
    node1 = Node(state=[("John", "PROPN")], parent=None, action=None, path_score=1)
    node2 = Node(state=[("will", "AUX")], parent=None, action=None, path_score=2)
    frontier.add(node1)
    frontier.add(node2)
    assert frontier.pop() == node1
    assert frontier.pop() == node2

def test_frontier_contains_phrase():
    """Test the contains_phrase method in the Frontier."""
    frontier = Frontier()
    node = Node(state=[("The", "DET"), ("project", "NOUN")], parent=None, action=None, path_score=0)
    assert frontier.contains_phrase(node, "NP") is True
    node = Node(state=[("will", "AUX"), ("complete", "VERB")], parent=None, action=None, path_score=0)
    assert frontier.contains_phrase(node, "VP") is True

def test_score_phrase(setup_organize_request):
    """Test the score_phrase method."""
    organize_request = setup_organize_request 
    phrase = ["complete", "the", "project"]
    score = organize_request.score_phrase(phrase, "Task")
    assert score > 0  # Ensure the score is calculated and is greater than 0

def test_search_by_category(setup_organize_request):
    """Test the search_by_category method."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    organize_request = setup_organize_request
    logger.info(f"Starting search_by_category test on {organize_request.request}.")
    
    results = organize_request.search_by_category()
    logger.info(f"search_by_category returned {results}.") 
    
    assert len(results) > 0  # Ensure results are returned
    logger.info("Results are not empty.")
    
    assert all(isinstance(node, Node) for node in results)  # Ensure all results are Node instances
    logger.info("All results are instances of Node.")

def test_fetch_others(setup_organize_request):
    """Test the fetch_others method."""
    organize_request = setup_organize_request
    phrases = organize_request.fetch_others()
    assert "DATE" in phrases
    assert "PERSON" in phrases
    assert "ORG" in phrases
    assert "John" in phrases["PERSON"]  # Ensure "John" is identified as a PERSON
    assert "Friday" in phrases["DATE"]  # Ensure "Friday" is identified as a DATE

def test_integration_task_and_person_identification():
    """Integration test for identifying tasks and assigned persons."""
    request = "Alice will review the design document by Monday."
    organize_request = OrganizeRequest(request, max_vision=5, phrase_type="NP", phrase_category="Task", beam_width=3)
    results = organize_request.search_by_category()
    assert len(results) > 0  # Ensure results are returned
    assert any("Alice" in node.state for node in results)  # Ensure "Alice" is identified
    assert any("design document" in " ".join(token[0] for token in node.state) for node in results)  # Ensure "design document" is identified

    phrases = organize_request.fetch_others()
    assert "Alice" in phrases["PERSON"]  # Ensure "Alice" is identified as a PERSON
    assert "Monday" in phrases["DATE"]  # Ensure "Monday" is identified as a DATE