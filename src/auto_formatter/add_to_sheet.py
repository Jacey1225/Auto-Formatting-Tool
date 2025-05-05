import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials


SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
credentials = Credentials.from_authorized_user_file("google_API/auto-formatting-api-23a0e2ed9b8c.json")

class AddToSheet:
    def __init__(self, request, sheet_id=None, range_name=None):
        self.request = request
        self.sheet_id = sheet_id
        self.range_name = range_name
    
    def fetch_entities(self):
        return