import spacy 
from transformers import BertTokenizer, BertModel
import torch  
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

class Node:
    def __init__(self, state, parent, action, path_score):
        self.state = state if state else []
        self.parent = parent
        self.action = action
        self.path_score = path_score

class Frontier:
    def __init__(self, beam_width=None):
        self.frontier = []
        self.beam_width = beam_width

        self.rules = {
            "NP": [
                ["DET", "NOUN"],
                ["DET", "ADJ", "NOUN"],
                ["DET", "ADJ", "ADJ", "NOUN"],
                ["ADJ", "NOUN"],
                ["ADJ", "ADJ", "NOUN"],
                ["ADJ", "ADJ", "ADJ", "NOUN"],
                ["ADJ", "ADJ", "ADJ", "ADJ", "NOUN"],
                ["NOUN", "ADP", "NOUN"],
                ["NOUN", "ADP", "ADJ", "NOUN"],
                ["NOUN", "ADP", "ADJ", "ADJ", "NOUN"],
                ["NOUN", "NOUN"],
                ["DET", "NOUN", "VERB", "DET", "NOUN"],
                ["DET", "NOUN", "SCONJ", "VERB"],
                ["NOUN", "ADP", "DET", "NOUN"]
            ],
            "VP": [
                ["VERB", "NOUN"],
                ["VERB", "ADJ", "NOUN"],
                ["VERB", "ADJ", "ADJ", "NOUN"],
                ["VERB", "ADJ", "ADJ", "ADJ", "NOUN"],
                ["AUX", "VERB"],
                ["AUX", "VERB", "NOUN"],
                ["AUX", "VERB", "ADJ", "NOUN"],
                ["AUX", "VERB", "ADJ", "ADJ", "NOUN"],
                ["VERB", "ADP", "DET", "NOUN"],
                ["VERB", "ADV"],
                ["PRON", "VERB", "PRON", "NOUN"],
                ["PRON", "VERB", "PROPN", "DET", "NOUN"],
                ["PRON", "VERB"],
                ["PRON", "VERB", "NOUN"],
                ["VERB", "DET", "NOUN"],
            ]
        }
    
    def invalid(self, max_vision):
        if len(self.frontier) < 1 or not self.frontier:
            return True
        return False

    def add(self, node):
        self.frontier.append(node)

    def pop(self): 
        if self.invalid(len(self.frontier)):
            raise Exception("Frontier is empty")
        else:
            node = self.frontier.pop(0)
            return node
        
    def get_nodes(self, node):
        return node.state[1:]
    
    def contains_phrase(self, node, phrase_type):
        """Check if the current set of nodes contains a phrase of the given type


        Args:
            nodes (list): list of the current nodes that have been taken from the fontier
            phrase_type (str): the type of phrase we are searching for (either NP or VP)

        Returns:
            bool: True if there is a valid pattern for the given node, False otherwise
        """
        if not node.state or len(node.state) == 0:
            return False

        if not isinstance(node.state[0], tuple):
            return False
        
        tag_sequence = [token[1] for token in node.state]
        for rule in self.rules[phrase_type]:
            for i in range(len(tag_sequence) - len(rule) + 1):
                if tag_sequence[i:i+len(rule)] == rule:
                    return True
        return False
    
    def prune(self, beam_width):
        self.frontier.sort(key=lambda node: node.path_score, reverse=True)
        return self.frontier[:beam_width]
            

#############################
# BEAM SEARCH PRUNING CLASS #
#############################

class EmbedTrainingData:
    def __init__(self):
        """embed all data within the training dataset --> used to perform a beam search on the nodes given within 
        within the current frontier. Remove top k candidates that do not fall in line with the training data as accurately as others

        Args:
            filename (str): path to training data
        """
        self.training_data = pd.read_csv("src/auto_formatter/data/TrainData.csv")
        self.training_data = self.training_data.dropna()
        self.training_data.drop(self.training_data[self.training_data["Responsible"] == "TPE"].index)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.model.eval()
    
    def embed_tasks(self, column_to_embed):
        """embed a column within the training data given the target column name, specifically the responsibility and task column
        Args:
            column_to_embed (str): column to be embedded

        Returns (list): list of all embedded tasks within the training data
        """
        column = self.training_data[column_to_embed].tolist()

        embeddings = []
        for task in column:
            if task:
                encoded = self.tokenizer(task, is_split_into_words=False, return_tensors='pt', padding=True, truncation=True)
                with torch.no_grad():
                    outputs = self.model(**encoded)
                    embeddings.append(outputs.last_hidden_state.mean(dim=1))
        if embeddings:
            return embeddings

    def write_embeddings(self, filename):
        """gathers all columns from the training data, embeds them, and creates a new csv file 
        with the processed data. This is the nsued to score the candidates and keep the top k that best
        fit the training date

        """
        embeddings = {
            "Task": None,
            "Responsible": None
        }

        columns = ["Task", "Responsible"]
        for column in columns:
            embeddings[column] = self.embed_tasks(column)
            
        torch.save(embeddings, filename)

#####################################################################################
# BEAM SEARCH ALGORITHM --> REMOVE TOP K CANDIDATES FROM EACH STATE IN THE FRONTIER #
#####################################################################################
class OrganizeRequest(EmbedTrainingData):
    def __init__(self, request, max_vision, phrase_type, phrase_category, beam_width=3):
        """Perform beam search on the given request -- 
        1. Tokenize request
        2. Update the frontier for each given set of nodes
        3. Evaluate the nodes within the frontier using the established rules
        4. prune k candidates from the list of possible phrases based on their similarity score to the training data
        5. Return the top k candidates
        6. Repeat until the frontier is empty or the max_vision is reached 
        7. Return the top k candidates for the entire request

        Args:
            request (str): request to be parsed
            max_vision (int): maximum number of nodes to be considered at each step
            phrase_type (str): type of phrase to be extracted (either NP or VP)
            beam_width (int, optional): Maximum number of candidates o be returned at each step. Defaults to 3.
        """
        super().__init__()
        self.nlp = spacy.load("en_core_web_lg")
        self.request = request
        self.max_vision = max_vision
        self.phrase_type = phrase_type
        self.phrase_category = phrase_category
        self.frontier = Frontier()
        self.beam_width = beam_width

        filename = os.path.join("src", "auto_formatter", "data", "EmbeddedTrainingData.pt")
        if not os.path.exists(filename):
            self.write_embeddings(filename)
        
        self.trained_data = torch.load(filename)


    def score_phrase(self, phrase, phrase_category):
        """using BertTokenizer and BertModel, score the first layer of given candidate nodes
        and return the top k candidates based on their similarity score to the training data

        Returns:
            top_k (list): top k candidates that best fit the training data
        """
        phrase_text = [token[0] for token in phrase]
        tokens = self.tokenizer(phrase_text, is_split_into_words=True, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**tokens)
            embeddings = outputs.last_hidden_state.mean(dim=1)

        score = 0
        for element in self.trained_data[phrase_category]:
            element = element.reshape(1, -1)
            similarity = torch.cosine_similarity(embeddings, element, dim = 1)
            score += similarity.item()
        
        return score / len(self.trained_data[phrase_category])

    
    def search_by_category(self):
        """Utilizing the previous classes (Node, Frontier, and EmbedTrainingData)
        we perform a beam search on the given request, iteratively updating the frontier 
        for each sequence of nodes --> Hybrid: Predict is the current sequence of nodes is a valid pattern 
        to extract a phrase,and then score them via a trained dataset on a given category of phrases. 
        Finally, return the top k candidates of each node as well as the last set of candidates remaining


        Returns:
            list: A list of the top k candidates that best fit the desired category of phrase
        """
        doc = self.nlp(self.request) #process request for tagging

        tokens = [(token.text, token.pos_) for token in doc] #tokenize request
        initial_state = tokens[:min(self.max_vision, len(tokens))] #get the first n words of the request

        #create the starting node with the initial state, and empty data
        starting_node = Node(
            initial_state,
            parent=None,
            action=None,
            path_score=0
        )
        self.frontier.add(starting_node) #add the node to the frontier
        pos = 0 #set the position to the end of the initial state
        logger.info(f"initial frontier: {self.frontier.frontier}")

        final_results = [] #list to store the final results of the search
        while pos < len(tokens) and not self.frontier.invalid(self.max_vision): #main loop that slides through the request
            node = self.frontier.pop() #get the next node from the frontier
            is_pattern = self.frontier.contains_phrase(node, self.phrase_type) #verify if the node contains a valid pattern

            if not is_pattern:
                new_pos = pos + 1 #if no patterns match, move to the next position
                if new_pos >= len(tokens): #if the position is past the end of the request, break the loop
                    break

                new_state = tokens[new_pos:new_pos + min(self.max_vision, len(tokens) - new_pos)] #get the next n words of the request
                new_node = Node(
                    new_state,
                    parent=node,
                    action=None,
                    path_score=node.path_score
                )
                self.frontier.add(new_node) #add the new node to the frontier
            else:# if a pattern is found, score it, and create a new node that follows after the sequence position
                score = self.score_phrase(node.state, self.phrase_category)
                state_length = len(node.state)
        
                new_pos = pos + state_length
                if new_pos >= len(tokens):
                    break
                new_state = tokens[new_pos: new_pos + min(self.max_vision, len(tokens) - new_pos)]
                new_node = Node(
                    new_state,
                    parent=node,
                    action=is_pattern,
                    path_score=node.path_score + score
                )
                self.frontier.add(new_node)
                final_results.append(new_node)

            pos += 1
            self.frontier.prune(self.beam_width)
        if final_results != []:
            final_results.sort(key=lambda node: node.path_score, reverse=True)

            return final_results[:self.beam_width]
        else:
            return []
        

    def fetch_others(self):
        doc = self.nlp(self.request) #process request for tagging
        phrases = {
            "DATE": [],
            "TIME": [],
            "PERSON": [],
            "ORG": []
        }

        for ent in doc.ents:
            if ent.label_ == "DATE" or ent.label_ == "TIME" or ent.label_ == "PERSON" or ent.label_ == "ORG":
                phrases[ent.label_].append(ent.text)
        
        return phrases