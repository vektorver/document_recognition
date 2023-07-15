import requests
import json
from datetime import datetime, timedelta
from dotenv import dotenv_values
from dateutil import parser

dotenv_config = dotenv_values(".env")


class RequestToServer(object):

    def __init__(self, server):
        self.server = server
        self.headers = {'Content-type': 'application/json',
                        'Accept': 'text/plain'}
        
    def create_user(self, tg_chat_id):
        url = self.server + f'/api/v1/user/{tg_chat_id}'
        response = requests.post(url, headers=self.headers)
        json_data = json.loads(response.text)
        return json_data, response.status_code
    
    def get_user(self, tg_chat_id):
        url = self.server + f'/api/v1/user/{tg_chat_id}'
        response = requests.get(url, headers=self.headers)
        json_data = json.loads(response.text)
        return json_data, response.status_code
    
    def set_state(self, tg_chat_id, status):
        url = self.server + f'/api/v1/user/{tg_chat_id}/status/{status}'
        response = requests.put(url, headers=self.headers)
        json_data = json.loads(response.text)
        return json_data, response.status_code 
    
    def predict(self, tg_chat_id, file):
        url = self.server + f'/api/v1/passport/predict/{tg_chat_id}'
        files = {'file': open(file, 'rb')}
        response = requests.post(url, files=files)
        json_data = json.loads(response.text)
        return json_data, response.status_code

    def get_excel(self, tg_chat_id):
        url = self.server + f'/api/v1/passports/excel/{tg_chat_id}'
        response = requests.get(url, headers=self.headers)
        json_data = json.loads(response.text)
        return json_data, response.status_code