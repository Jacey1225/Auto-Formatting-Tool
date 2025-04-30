import spacy 
from transformers import BertTokenizer, BertModel
import torch  
import pandas as pd
import os

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
                ["DET", "NOUN", "NOUN"],
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
                ["PROPN", "VERB", "PROPN", "DET", "NOUN"],
                ["PROPN", "VERB"],
                ["PROPN", "VERB", "NOUN"],
                ["VERB", "DET", "NOUN"],
            ]
        }
    
    def invalid(self, max_vision):
        if len(self.frontier) < max_vision or not self.frontier:
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
            list: a list of the nodes in sequence as to how they match to the phrase type
        """
        if not node.state or len(node.state) == 0:
            return []

        if not isinstance(node.state[0], tuple) or len(node.state[0]) < 2:
            return []
        
        first_tag = node.state[0][1]
        first_text = node.state[0][0]
        children = self.get_nodes(node)

        matches = []

    
        for rule in self.rules[phrase_type]:
            sequence = []
            isValid = True
            if first_tag == rule[0]:
                i = 1
                sequence.append(first_text)
                while isValid and i < len(children) and i < len(rule):
                    if children[i][1] == rule[i]:
                        sequence.append(children[i][0])
                        i += 1
                    else:
                        isValid = False
                        break
                if isValid:
                    matches.append(sequence)
            else:
                continue
        return matches
    
    def prune(self, beam_width):
        self.frontier.sort(key=lambda node: node.path_score, reverse=True)
        return self.frontier[:beam_width]
            

#############################
# BEAM SEARCH PRUNING CLASS #
#############################

class EmbedTrainingData:
    def __init__(self, filename):
        """embed all data within the training dataset --> used to perform a beam search on the nodes given within 
        within the current frontier. Remove top k candidates that do not fall in line with the training data as accurately as others

        Args:
            filename (str): path to training data
        """
        self.training_data = pd.read_csv(filename)
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
                    embeddings.append(outputs.last_hidden_state[0])
        if embeddings:
            return embeddings

    def write_embeddings(self):
        """gathers all columns from the training data, embeds them, and creates a new csv file 
        with the processed data. This is the nsued to score the candidates and keep the top k that best
        fit the training date

        """
        filename = "auto_formatter/data/embedded_training_data.csv"
        
        embeddings = {
            "Task": None,
            "Responsible": None
        }

        columns = ["Task", "Responsible"]
        for column in columns:
            embeddings[column] = self.embed_tasks(column)
            
        df = pd.DataFrame(embeddings, columns=columns)
        df.to_csv(filename, index=False)

#####################################################################################
# BEAM SEARCH ALGORITHM --> REMOVE TOP k CANDIDATES FROM EACH STATE IN THE FRONTIER #
#####################################################################################
class OrganizeRequest(EmbedTrainingData):
    def __init__(self, request, max_vision, phrase_type, phrase_category, filename, beam_width=3):
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
        super().__init__(filename)
        self.nlp = spacy.load("en_core_web_lg")
        self.request = request
        self.max_vision = max_vision
        self.phrase_type = phrase_type
        self.phrase_category = phrase_category
        self.frontier = Frontier()
        self.beam_width = beam_width

        filename = "auto_formatter/data/embedded_training_data.csv"
        if not os.path.exists(filename):
            self.write_embeddings()
        
        self.trained_data = pd.read_csv(filename)
        self.trained_data = self.trained_data.dropna()
        self.trained_data.drop(self.trained_data[self.trained_data["Responsible"] == "TPE"].index)


    def score_phrase(self, phrase, phrase_category):
        """using BertTokenizer and BertModel, score the first layer of given candidate nodes
        and return the top k candidates based on their similarity score to the training data

        Returns:
            top_k (list): top k candidates that best fit the training data
        """
        tokens = self.tokenizer(phrase, is_split_into_words=True, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**tokens)
            embeddings = outputs.last_hidden_state[0]

        score = 0
        for element in self.trained_data[phrase_category]:
            element = element.reshape(1, -1)
            score = torch.cosine_similarity(embeddings.reshape(1, -1), element, dim = 1)
            score += score
        return score / len(self.trained_data[phrase_category])

    
    def search_by_category(self):
        """Utilizing the previous classes (Node, Frontier, and EmbedTrainingData)
        we perform a beam search on the given request, iteratively updating the frontier 
        for each sequence of nodes --> Hybrid: Predict the possible sequences of nodes to extract a phrase
        and then score them via atrained dataset on a given category of phrases. 
        Finally, return the top k candidates of each node as well as the last set of candidates remaining


        Returns:
            list: A list of the top k candidates that best fit the desired category of phrase
        """
        doc = self.nlp(self.request) #process request for tagging

        tokens = [(token.text, token.tag_) for token in doc] #tokenize request
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

        final_results = [] #list to store the final results of the search
        while pos < len(tokens): #main loop that slides through the request
            node = self.frontier.pop() #get the next node from the frontier
            next_frontier = Frontier(self.beam_width) #create a new frontier
            patterns = self.frontier.contains_phrase(node, self.phrase_type) #get the patterns that match the node
            
            if patterns == []:
                pos += 1 #if no patterns match, move to the next position
                if pos >= len(tokens): #if the position is past the end of the request, break the loop
                    break

                new_state = tokens[pos:pos + min(self.max_vision, len(tokens) - pos)] #get the next n words of the request
                new_node = Node(
                    new_state,
                    parent=node,
                    action=None,
                    path_score=node.path_score
                )
                next_frontier.add(new_node) #add the new node to the frontier
            else:   
                for pattern in patterns:
                    score = self.score_phrase(pattern, self.phrase_category)
                    pattern_length = len(pattern)
                    if pattern_length + pos > len(tokens):
                        pattern_length = 1
                    
                    new_pos = pos + pattern_length
                    if new_pos >= len(tokens):
                        new_state = []
                    else:
                        new_state = tokens[new_pos: new_pos + min(self.max_vision, len(tokens) - new_pos)]

                    new_node = Node(
                        new_state,
                        parent=node,
                        action=pattern,
                        path_score=node.path_score + score
                    )
                    next_frontier.add(new_node)
                    final_results.append(new_node)

            next_frontier.prune(self.beam_width)
            self.frontier = next_frontier
            pos += 1
        if final_results != []:
            final_results.sort(key=lambda node: node.path_score, reverse=True)
            return final_results[:self.beam_width]
        else:
            return []

    