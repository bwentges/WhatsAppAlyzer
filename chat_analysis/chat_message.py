from __future__ import annotations
from typing import List, Optional

from itertools import cycle

import matplotlib.pyplot as plt

import datetime

# Define a pattern for line styles, e.g., "-" 10 times, ":" 10 times
line_style_pattern = ["-"] * 10 + [":"] * 10  # Adjust the pattern as needed
LINE_STYLES = cycle(line_style_pattern)
COLORS = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])


class ChatMessage:
    """A class to represent a message in a chat conversation."""
    def __init__(self, timestamp: datetime, sender: ChatSender, content: str):
        self.timestamp: datetime = timestamp
        self.sender: ChatSender = sender
        self.content: str = content
        self.word_count: int = self.get_word_count()
        self.next_message: Optional[ChatMessage] = None
        self.previous_message: Optional[ChatMessage] = None

    def __repr__(self):
        return f"{self.timestamp} - {self.sender.name}: {self.content}"

    def get_word_count(self):
        return len(self.content.split(" "))


class ChatSender:
    """A class to represent a sender in a chat conversation."""
    all_senders: List[ChatSender] = []
    index_to_name = {}

    def __init__(self, name: str):
        self.name: str = name
        self.messages: List[ChatMessage] = []
        self.line_style = next(LINE_STYLES)
        self.color = next(COLORS)
        self.index = len(ChatSender.all_senders)
        ChatSender.index_to_name[self.index] = self.name
        ChatSender.all_senders.append(self)

    def add_message(self, message: ChatMessage):
        self.messages.append(message)

    def message_count(self) -> int:
        return len(self.messages)

    @staticmethod
    def get_all_sender_color_mapping():
        return {sender.name: sender.color for sender in ChatSender.all_senders}