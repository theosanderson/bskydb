import nltk
from nltk.corpus import cmudict
import itertools
import re
from collections import defaultdict
from functools import lru_cache
import json
import asyncio
import websockets
import psycopg2
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Download required NLTK data
nltk.download('cmudict', quiet=True)
pronouncing_dict = cmudict.dict()

# Create word stress patterns cache
word_stress_patterns = defaultdict(list)
for word, pronunciations in pronouncing_dict.items():
    word_stress_patterns[word] = [
        [int(phoneme[-1]) for phoneme in pron if phoneme[-1] in '012']
        for pron in pronunciations
    ]

# Text processing utilities
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

def is_iambic_pentameter(pattern):
    """Check if a stress pattern is iambic pentameter."""
    return len(pattern) == 10 and all(
        (s == 0 or s == 2) if i % 2 == 0 else s == 1
        for i, s in enumerate(pattern)
    )

@lru_cache(maxsize=10000)
def check_iambic_pentameter(text):
    """Check if text is in iambic pentameter."""
    if len(text.split()) < 5 or len(text.split()) > 75:
        return False
    
    if any(b in text for b in banned):
        return False
    
    for k, v in substitutions.items():
        text = text.replace(k, v)
    
    text = numerals_to_words(text)
    text = re.sub(r'[^A-Za-z\s\']', '', text.lower())
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    
    if len(set(words)) == 1:
        return False
    
    if not all(word in word_stress_patterns for word in words):
        return False
    
    stress_options = [word_stress_patterns[word] for word in words]
    stress_options = [[s[0]] for s in stress_options]
    
    if sum(len(p) > 1 for p in stress_options) > 3:
        return False    
    
    min_syllables = sum(min(len(patterns) for patterns in p) for p in stress_options)
    max_syllables = sum(max(len(patterns) for patterns in p) for p in stress_options)
    if min_syllables > 10 or max_syllables < 10:
        return False
    
    for stress_combination in itertools.product(*stress_options):
        pattern = [s for stresses in stress_combination for s in stresses]
        if is_iambic_pentameter(pattern):
            return True
    return False

class IambicMonitor:
    def __init__(self):
        self.conn = None
        self.cur = None
        
    async def init_db(self):
        """Initialize database connection and create table if it doesn't exist."""
        try:
            self.conn = psycopg2.connect(
                dbname="postgres",
                user='postgres',
                password="dev-password-123",
                host='localhost',
                port=5432
            )
            self.cur = self.conn.cursor()
            
            # Create table if it doesn't exist
            create_table_query = """
            CREATE TABLE IF NOT EXISTS iambic_messages (
                id SERIAL PRIMARY KEY,
                message_id TEXT NOT NULL UNIQUE,
                did TEXT NOT NULL,
                post_text TEXT NOT NULL,
                timestamp_us BIGINT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                added_to_poem_at TIMESTAMP
            );
            """
            
            # Create indices
            create_indices = """
            CREATE INDEX IF NOT EXISTS idx_iambic_timestamp ON iambic_messages(created_at);
            CREATE INDEX IF NOT EXISTS idx_iambic_message_id ON iambic_messages(message_id);
            """
            
            self.cur.execute(create_table_query)
            self.cur.execute(create_indices)
            self.conn.commit()
            
            logging.info("Successfully initialized database and created table")
        except Exception as e:
            logging.error(f"Database initialization error: {e}")
            if self.conn:
                self.conn.rollback()
            raise

    async def insert_iambic_post(self, message_id, did, text, timestamp_us):
        """Insert an iambic post into the database."""
        try:
            query = """
                INSERT INTO iambic_messages (message_id, did, post_text, timestamp_us)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (message_id) DO NOTHING
            """
            self.cur.execute(query, (message_id, did, text, timestamp_us))
            self.conn.commit()
            logging.info(f"Found and inserted iambic post: {text}")
            print(f"Found and inserted iambic post: {text}")
        except Exception as e:
            logging.error(f"Error inserting post {message_id}: {e}")
            self.conn.rollback()

    async def process_message(self, message):
        """Process a message from the firehose."""
        try:
            if (message.get('commit', {}).get('record', {}).get('text') and 
                message['commit'].get('operation') == 'create' and 
                message['commit'].get('collection') == 'app.bsky.feed.post'):
                
                text = message['commit']['record']['text']
                
                if check_iambic_pentameter(text):
                    await self.insert_iambic_post(
                        message['commit']['rkey'],
                        message['did'],
                        text,
                        message['time_us']
                    )
            
        except Exception as e:
            logging.error(f"Error processing message: {e}")

    async def monitor_firehose(self):
        """Connect to and monitor the Bluesky firehose."""
        while True:
            try:
                async with websockets.connect('wss://bsky-relay.c.theo.io/subscribe?wantedCollections=app.bsky.feed.post') as websocket:
                    logging.info("Connected to Bluesky firehose")
                    
                    while True:
                        message = await websocket.recv()
                        await self.process_message(json.loads(message))
                        
            except Exception as e:
                logging.error(f"WebSocket error: {e}")
                await asyncio.sleep(5)
                logging.info("Attempting to reconnect...")

    async def run(self):
        """Main run function."""
        await self.init_db()
        await self.monitor_firehose()

if __name__ == "__main__":
    monitor = IambicMonitor()
    asyncio.run(monitor.run())