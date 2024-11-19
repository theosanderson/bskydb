import psycopg2
from contextlib import contextmanager
import os
import logging
from datetime import datetime
import nltk
from nltk.corpus import cmudict
from collections import defaultdict
import itertools
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Download and initialize CMU dict
nltk.download('cmudict', quiet=True)
pronouncing_dict = cmudict.dict()

# Reuse the text processing utilities from the monitor
substitutions = {
    '&': ' and ',
    'w/': 'with',
    'w/o': 'without',
}

banned = ['$', '#', '@', '/']

def num2words(num):
    """Convert number to words."""
    under_20 = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven',
                'Eight', 'Nine', 'Ten', 'Eleven', 'Twelve', 'Thirteen', 'Fourteen',
                'Fifteen', 'Sixteen', 'Seventeen', 'Eighteen', 'Nineteen']
    tens = ['Twenty', 'Thirty', 'Forty', 'Fifty', 'Sixty', 'Seventy', 'Eighty', 'Ninety']
    above_100 = {100: 'Hundred', 1000: 'Thousand', 1000000: 'Million', 1000000000: 'Billion'}

    if num < 20:
        return under_20[num]
    if num < 100:
        return tens[(num // 10) - 2] + ('' if num % 10 == 0 else ' ' + under_20[num % 10])
    pivot = max([key for key in above_100.keys() if key <= num])
    return num2words(num // pivot) + ' ' + above_100[pivot] + ('' if num % pivot == 0 else ' ' + num2words(num % pivot))

def numerals_to_words(text):
    """Convert numerals to words in text."""
    return re.sub(r'(\d+)', lambda m: num2words(int(m.group(0))), text)

def get_last_phoneme(word):
    """Get the last phoneme sequence for a word."""
    if word.lower() not in pronouncing_dict:
        return None
    # Get the first pronunciation (most common)
    pronunciation = pronouncing_dict[word.lower()][0]
    # Find the last stressed vowel and return everything after it
    for i, phoneme in enumerate(reversed(pronunciation)):
        if any(char.isdigit() for char in phoneme):
            return tuple(pronunciation[-i-1:])
    return tuple(pronunciation)

def do_words_rhyme(word1, word2):
    """Check if two words rhyme."""
    if word1.lower() == word2.lower():
        return False
    
    phoneme1 = get_last_phoneme(word1)
    phoneme2 = get_last_phoneme(word2)
    
    if not phoneme1 or not phoneme2:
        return False
        
    return phoneme1 == phoneme2

def get_last_word(text):
    """Get the last word from a line of text."""
    text = numerals_to_words(text)
    text = re.sub(r'[^A-Za-z\s\']', '', text.lower())
    words = text.split()
    return words[-1] if words else None

class IambicRhymeFinder:
    def __init__(self):
        self.db_params = {
            'dbname': 'bluesky_posts',
            'user': 'postgres',
            'password': 'dev-password-123',
            'host': 'localhost',
            'port': 5432
        }

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = None
        try:
            conn = psycopg2.connect(**self.db_params)
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def get_last_repost_timestamp(self):
        """Get the timestamp of the most recent reposted message."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                query = """
                    SELECT created_at
                    FROM iambic_messages
                    WHERE reposted_at IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT 1
                """
                cur.execute(query)
                result = cur.fetchone()
                return result[0] if result else datetime.min

    def get_new_messages(self, last_timestamp):
        """Get all messages since the last reposted timestamp."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                query = """
                    SELECT id, message_id, post_text, created_at
                    FROM iambic_messages
                    WHERE created_at > %s
                    ORDER BY created_at ASC
                """
                cur.execute(query, (last_timestamp,))
                return cur.fetchall()

    def find_rhyming_couplets(self):
        """Main function to find rhyming couplets."""
        # Get the last repost timestamp
        last_timestamp = self.get_last_repost_timestamp()
        logging.info(f"Last repost timestamp: {last_timestamp}")

        # Get all new messages
        messages = self.get_new_messages(last_timestamp)
        logging.info(f"Found {len(messages)} new messages to analyze")

        # Group messages by their end rhyme
        rhyme_groups = defaultdict(list)
        
        for msg_id, message_id, text, created_at in messages:
            last_word = get_last_word(text)
            if last_word:
                last_phoneme = get_last_phoneme(last_word)
                if last_phoneme:
                    rhyme_groups[last_phoneme].append((msg_id, message_id, text, created_at))

        # Find rhyming couplets
        couplets = []
        for rhyme_group in rhyme_groups.values():
            if len(rhyme_group) < 2:
                continue
                
            for msg1, msg2 in itertools.combinations(rhyme_group, 2):
                # Check if the messages are temporally close (within 24 hours)
                if abs((msg1[3] - msg2[3]).total_seconds()) <= 86400:
                    couplets.append((msg1, msg2))

        # Sort couplets by timestamp
        couplets.sort(key=lambda x: min(x[0][3], x[1][3]))

        return couplets

if __name__ == "__main__":
    finder = IambicRhymeFinder()
    couplets = finder.find_rhyming_couplets()
    
    print(f"\nFound {len(couplets)} potential rhyming couplets:")
    for (id1, msg_id1, text1, time1), (id2, msg_id2, text2, time2) in couplets:
        print("\nCouplet:")
        print(f"1: {text1}")
        print(f"2: {text2}")
        print(f"Posted: {time1} and {time2}")
        print(f"Message IDs: {msg_id1} and {msg_id2}")