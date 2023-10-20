from __future__ import annotations

import matplotlib.pyplot as plt

from collections import defaultdict

import numpy as np

from chat_analysis.chat_message import ChatSender, ChatMessage

from typing import List, Dict


def plot_consecutive_message_counts(consecutive_message_counts: Dict[ChatSender, list[int]]):
    for sender, counts in consecutive_message_counts.items():
        plt.figure()
        plt.hist(counts, bins=range(1, max(counts) + 2), align='left', alpha=0.5, color='b')
        plt.title(f"{sender.name} - Consecutive Message Counts")
        plt.xlabel("Consecutive Message Counts")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()


def plot_consecutive_message_counts_2(consecutive_message_counts: Dict[ChatSender, list[int]], threshold=5):
    fig, ax = plt.subplots()

    for sender, counts in consecutive_message_counts.items():
        counts_above_threshold = [count for count in counts if count > threshold]

        if counts_above_threshold:
            ax.hist(counts_above_threshold, bins=range(threshold + 1, max(counts_above_threshold) + 2),
                    align='left', alpha=0.5, label=sender.name, histtype="step", stacked=True, fill=False)
            # ax.hist(counts_above_threshold, bins=20, density=True,
            #         align='left', alpha=0.5, label=sender.name, histtype="bar")

    ax.set_title(f"Consecutive Message Counts (Counts > {threshold})")
    ax.set_xlabel("Consecutive Message Counts")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True)
    plt.show()


def plot_cumulative_messages(messages: List[ChatMessage]):
    plt.figure(figsize=(12, 6))
    # Count messages and update cumulative counts
    message_count = defaultdict(int)
    cumulative_counts = defaultdict(list)
    for message in messages:
        message_count[message.sender] += 1
        cumulative_counts[message.sender].append((message.timestamp, message_count[message.sender]))

    # Create a plot for each sender
    for sender, data in cumulative_counts.items():
        timestamps, counts = zip(*data)
        plt.plot(timestamps, counts, label=sender.name, linestyle=sender.line_style, color=sender.color)
        print(
            f"Average words per message for {sender.name}: {np.mean([message.word_count for message in sender.messages])}")

    # Customize the plot
    plt.xlabel('Date and Time')
    plt.ylabel('Cumulative Messages')
    plt.title('Cumulative Messages Over Time')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Display the plot
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_cumulative_words(messages: List[ChatMessage]):
    plt.figure(figsize=(12, 6))
    # Count messages and update cumulative counts
    message_count = defaultdict(int)
    cumulative_counts = defaultdict(list)
    for message in messages:
        message_count[message.sender] += message.get_word_count()
        cumulative_counts[message.sender].append((message.timestamp, message_count[message.sender]))

    # Create a plot for each sender
    for sender, data in cumulative_counts.items():
        timestamps, counts = zip(*data)
        plt.plot(timestamps, counts, label=sender.name, linestyle=sender.line_style, color=sender.color)

    # Customize the plot
    plt.xlabel('Date and Time')
    plt.ylabel('Cumulative Words')
    plt.title('Cumulative Words Over Time')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Display the plot
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_messages_per_day(self):
    plt.figure(figsize=(12, 6))

    message_counts = defaultdict(lambda: defaultdict(int))
    for message in self.messages:
        day = message.timestamp.date()
        year = message.timestamp.year
        month = message.timestamp.month

        # Use a tuple (year, month) as the key to represent each month
        # day = (year, month)
        # Use a numerical index to represent each month
        # day = year * 12 + month

        # day = message.timestamp.isocalendar()[1]
        user = message.sender.name
        message_counts[day][user] += 1

    # Convert the message_counts dictionary to a 2D array or DataFrame
    users = sorted(set(user for day_data in message_counts.values() for user in day_data))
    days = sorted(message_counts.keys())

    # Create a 2D array of message counts
    message_count_array = np.zeros((len(days), len(users)))

    for i, day in enumerate(days):
        for j, user in enumerate(users):
            if user in message_counts[day]:
                message_count_array[i][j] = message_counts[day][user]

    # Create a DataFrame from the 2D array of message counts
    plt.stackplot(days, message_count_array.T, labels=users)

    plt.xlabel('Date')
    plt.ylabel('Cumulative Message Count')
    plt.xticks(rotation=45)
    plt.title('Cumulative Number of Messages per Day per User (Stacked Line Plot)')

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Display the plot
    plt.tight_layout()
    plt.show()

    # Sum the message counts for each day
    total_message_counts = message_count_array.sum(axis=1)

    # Create a list of (day, total_message_count) tuples
    day_message_count_pairs = [(day, total_count) for day, total_count in zip(days, total_message_counts)]

    # Sort the pairs by total_message_count in descending order
    sorted_day_message_count_pairs = sorted(day_message_count_pairs, key=lambda x: x[1], reverse=True)

    # Get the top 10 days
    top_10_days = sorted_day_message_count_pairs[:10]

    # Print the top 10 days
    for day, total_count in top_10_days:
        print(f"Day: {day}, Total Message Count: {int(total_count)}")


def plot_sender_interaction_heatmap(messages: List[ChatMessage], exclude_self=True, normalize_by_sender=False,
                                    normalize_by_next_sender=False):
    relation = np.zeros((len(ChatSender.all_senders), len(ChatSender.all_senders)))

    for message in messages[0:-2]:
        sender = message.sender.index
        next_sender = message.next_message.sender.index
        relation[sender][next_sender] += 1

    if exclude_self:
        for i in range(len(ChatSender.all_senders)):
            relation[i][i] = 0

    if normalize_by_sender:
        # Make percentage
        row_sums = relation.sum(axis=1) / 100

        # Divide each row by its sum using broadcasting
        relation = relation / row_sums[:, np.newaxis]

    if normalize_by_next_sender:
        # Make percentage
        col_sums = relation.sum(axis=0) / 100

        # Divide each column by its sum using broadcasting
        relation = relation / col_sums[np.newaxis, :]

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(relation, cmap="coolwarm", interpolation='nearest')

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(ChatSender.all_senders)), labels=ChatSender.index_to_name.values())
    ax.set_yticks(np.arange(len(ChatSender.all_senders)), labels=ChatSender.index_to_name.values())

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(ChatSender.all_senders)):
        for j in range(len(ChatSender.all_senders)):
            if i == j and exclude_self:
                continue
            text = ax.text(j, i, int(relation[i, j]),
                           ha="center", va="center", color="black", fontsize=12)

    ax.set_title("Sender Interaction Heatmap")
    if normalize_by_sender:
        ax.set_title("Sender Interaction Heatmap (Normalized by Sender)")
    if normalize_by_next_sender:
        ax.set_title("Sender Interaction Heatmap (Normalized by Next Sender)")
    ax.set_xlabel("Next Sender")
    ax.set_ylabel("Sender")
    fig.tight_layout()
    plt.show()