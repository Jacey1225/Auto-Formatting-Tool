from src.auto_formatter.organize_requests import OrganizeRequest

class Main:
    def __init__(self, request, filename):
        self.input = request
        self.filename = filename
        self.organize = None
    

    def identify_entities(self):
        verb_key = "VB"
        verb_category = "Task"

        noun_key = "NP"
        noun_category = "Responsible"

        self.organize_verbs = OrganizeRequest(self.input, max_vision=3, phrase_type=verb_key, phrase_category=verb_category)
        self.organize_nouns = OrganizeRequest(self.input, max_vision=3, phrase_type=noun_key, phrase_category=noun_category)

        task_entities = self.organize_verbs.search_by_category()
        responsible_entities = self.organize_nouns.search_by_category()

        return task_entities, responsible_entities
