# chat_history.py
from chat_analysis.chat_message import ChatSender, ChatMessage
import re
import os
from typing import List, Dict
import json
from datetime import datetime

from collections import defaultdict

from nltk.corpus import stopwords

import pprint


class ChatHistory:
    """This class is used to store, load, and manipulate the chat history"""
    def __init__(self, chat_language="dutch"):
        self.messages: List[ChatMessage] = []
        self.senders: Dict[str, ChatSender] = {}
        self.consecutive_message_counts: Dict[ChatSender, list[int]] = {}
        self.consecutive_messages_history: Dict[ChatSender, List[List[ChatMessage]]] = {}
        self.sender_mappings: {} = None
        self.stop_words = set(stopwords.words(chat_language))

    def load_sender_mappings(self, location: str = "data/sender_mapping.json"):
        """Load a JSON file containing sender mappings in order to merge senders with different names or change
        the names of senders.
        File format of sender_mapping.json:
        {
            "old_sender_name_1": "new_sender_name_1",
            "old_sender_name_2": "new_sender_name_2",
        }
        """
        try:
            with open(location, "r", encoding='utf-8') as file:
                self.sender_mappings = json.load(file)
            # return sender_mappings
        except (FileNotFoundError, json.JSONDecodeError):
            print("No sender mappings found.")

    def map_sender(self, sender):
        if self.sender_mappings is None:
            return sender
        if sender in self.sender_mappings:
            return self.sender_mappings[sender]
        return sender

    def add_message(self, timestamp: datetime, sender: str, content: str):
        sender = self.map_sender(sender)

        # Define a regular expression pattern to match any digit (0-9)
        pattern = r'\d'

        # Use the search() function to find the first match in the input string
        match = re.search(pattern, sender)

        if match:
            return  # Skip numbers

        if sender not in self.senders:
            self.senders[sender] = ChatSender(sender)
        message = ChatMessage(timestamp, self.senders[sender], content)

        self.messages.append(message)
        self.senders[sender].add_message(message)

    def load_chat_from_directory(self, directory_path):
        # Check if the directory path is valid
        if not os.path.isdir(directory_path):
            raise Exception("Invalid directory path")

        # Retrieve a list of all files in the directory
        file_paths = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path)]
        self.load_chat_from_files(file_paths)

    def load_chat_from_files(self, file_paths: List[str] or str):

        if isinstance(file_paths, str):
            file_paths = [file_paths]

        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as file:
                message_pattern = r'(\d{2}-\d{2}-\d{4} \d{2}:\d{2}) - ([^:]+): (.*)'
                lines = file.readlines()

                for line in lines:
                    match = re.match(message_pattern, line)

                    if match:
                        timestamp = datetime.strptime(match.group(1), '%d-%m-%Y %H:%M')
                        sender = match.group(2)
                        content = match.group(3)

                        if "<Media weggelaten>" in content:
                            continue    # Skip media messages

                        self.add_message(timestamp, sender, content)

        # Sort based on timestamps, as the files are not necessary in order
        self.sort_messages_by_timestamp()
        self.link_messages()

    def get_total_message_count(self):
        return len(self.messages)

    def get_most_active_sender(self):
        sender_counts = {}
        for message in self.messages:
            sender = message.sender
            sender_counts[sender] = sender_counts.get(sender, 0) + 1
        return max(sender_counts, key=sender_counts.get)

    def sort_messages_by_timestamp(self):
        """Sort the messages by timestamp"""
        self.messages.sort(key=lambda x: x.timestamp)

    def link_messages(self):
        """Link messages to each other, so that you can easily find the next and previous message of a message"""
        for i in range(0, len(self.messages) - 1):
            self.messages[i].next_message = self.messages[i + 1]
            self.messages[i + 1].previous_message = self.messages[i]

    def extend_stop_words(self, words: List[str]):
        """Add words to the stop words set, if you think they should be excluded from the word count"""
        self.stop_words.update(words)

    def count_words_per_sender(self):
        # Create a set of stopwords
        self.extend_stop_words(["wel", "we", "weer", "gaan", "jullie", "jij", "even", "mee", "nee", "echt", "ga",
                           "kom", "gaat", "wij"])

        # Create a dictionary to store word counts per sender
        word_counts = defaultdict(lambda: defaultdict(int))
        sum_word_counts = defaultdict(int)

        # Iterate through chat messages and count words per sender
        for message in self.messages:
            sender_name = message.sender.name
            words = message.content.split()
            for word in words:
                # Exclude stopwords and convert words to lowercase
                if word.lower() not in self.stop_words:
                    word_counts[sender_name][word.lower()] += 1
                    sum_word_counts[word.lower()] += 1


        # Find the most popular words for each sender
        most_popular_words = {}
        for sender_name, word_count_dict in word_counts.items():
            sorted_words = sorted(word_count_dict.items(), key=lambda x: x[1], reverse=True)
            most_popular_words[sender_name] = sorted_words

        most_popular_words["sum"] = sorted(sum_word_counts.items(), key=lambda x: x[1], reverse=True)

        return most_popular_words

    def get_consecutive_messages(self):
        counter = 1
        previous_sender = None
        message_chain = []
        for message in self.messages:
            sender = message.sender

            if sender not in self.consecutive_message_counts:
                self.consecutive_message_counts[sender] = []

            if sender not in self.consecutive_messages_history:
                self.consecutive_messages_history[sender] = []

            if sender == previous_sender:
                counter += 1
                message_chain.append(message)
            else:
                if previous_sender is not None:
                    self.consecutive_message_counts[previous_sender].append(counter)
                    self.consecutive_messages_history[previous_sender].append(message_chain)
                counter = 1
                message_chain = [message]

            previous_sender = sender

        for sender, counts in self.consecutive_message_counts.items():
            paired_list = list(zip(counts, self.consecutive_messages_history[sender]))
            paired_list.sort(key=lambda x: x[0], reverse=True)
            counts.sort(reverse=True)
            print(f"{sender.name}: {counts[0:10]}")
            pprint.pprint(f"{sender.name}:")
            pprint.pprint(paired_list[0:10])