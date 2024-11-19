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
from atproto import Client, models
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Iterator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Download and initialize CMU dict
nltk.download('cmudict', quiet=True)
pronouncing_dict = cmudict.dict()

@dataclass
class BlueskyPost:
    """Represents a post from the database"""
    id: int
    message_id: str
    post_text: str
    created_at: datetime
    did: str
    reposted_at: Optional[datetime] = None
    
    @property
    def uri(self) -> str:
        """Generate the at:// URI for the post"""
        feed_type = "app.bsky.feed.post"
        result= f"at://{self.did}/{feed_type}/{self.message_id}"
        
        return result
        
    @property
    def last_word(self) -> Optional[str]:
        """Get the last word of the post text"""
        return get_last_word(self.post_text)

@dataclass
class RhymingCouplet:
    """Represents a pair of rhyming posts"""
    first_post: BlueskyPost
    second_post: BlueskyPost
    
    @property
    def time_difference(self) -> float:
        """Get time difference between posts in seconds"""
        return abs((self.first_post.created_at - self.second_post.created_at).total_seconds())
    
    @property
    def most_recent_timestamp(self) -> datetime:
        """Get the timestamp of the most recent post in the couplet"""
        return max(self.first_post.created_at, self.second_post.created_at)

def get_last_phoneme(word):
    """Get the last phoneme sequence for a word."""
    if word.lower() not in pronouncing_dict:
        return None
    pronunciation = pronouncing_dict[word.lower()][0]
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
        self.client = Client()
        # Login using environment variables
        self.client.login(os.getenv('BSKY_HANDLE'), os.getenv('BSKY_PASSWORD'))

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

    def get_last_repost_timestamp(self) -> datetime:
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

    def get_new_messages(self, last_timestamp: datetime) -> Iterator[BlueskyPost]:
        """Get all messages since the last reposted timestamp as BlueskyPost objects."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                query = """
                    SELECT id, message_id, post_text, created_at, did
                    FROM iambic_messages
                    WHERE created_at > %s
                    ORDER BY created_at ASC
                """
                cur.execute(query, (last_timestamp,))
                for row in cur.fetchall():
                    yield BlueskyPost(
                        id=row[0],
                        message_id=row[1],
                        post_text=row[2],
                        created_at=row[3],
                        did=row[4]
                    )

    def is_valid_post(self, post: models.AppBskyFeedDefs.PostView) -> bool:
        """
        Check if a post is valid for our purposes:
        - Not a repost
        - No images
        - Not deleted
        - No mentions
        - No link cards
        - No links in text
        """
        logging.info(post.indexed_at)
        logging.info(post.record.created_at)
        # if created at at indexed at are more than 1 day apart in either direction, skip (they are strs like 2024-11-19T17:46:18.552Z)
        indexed_at_unix_timestamp = datetime.strptime(post.indexed_at, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()
        created_at_unix_timestamp = datetime.strptime(post.record.created_at, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()
        if abs(indexed_at_unix_timestamp - created_at_unix_timestamp) > 86400:
            return False
        if post.embed != None:
            return False
        
        return True

    def fetch_valid_posts(self, posts: List[BlueskyPost]) -> List[BlueskyPost]:
        """Fetch and validate posts from Bluesky API"""
        valid_posts = []
        batch_size = 25
        
        # Create URI to post mapping
        uri_to_post = {post.uri: post for post in posts}
        uris = list(uri_to_post.keys())
        
        for i in range(0, len(uris), batch_size):
            batch_uris = uris[i:i + batch_size]
            try:
                api_posts = self.client.app.bsky.feed.get_posts({'uris': batch_uris})
                
                for api_post in api_posts.posts:
                    if self.is_valid_post(api_post):
                        if api_post.uri in uri_to_post:
                            valid_posts.append(uri_to_post[api_post.uri])
                    else:
                        logging.info(f"Filtered out post {api_post.uri}")
                        
            except Exception as e:
                logging.error(f"Error fetching batch of posts: {e}")
                continue
                
        return valid_posts

    def find_rhyming_couplets(self) -> List[RhymingCouplet]:
        """Main function to find rhyming couplets."""
        # Get the last repost timestamp
        last_timestamp = self.get_last_repost_timestamp()
        logging.info(f"Last repost timestamp: {last_timestamp}")

        # Get all new messages
        messages = list(self.get_new_messages(last_timestamp))
        logging.info(f"Found {len(messages)} new messages to analyze")

        # Fetch and validate posts
        valid_messages = self.fetch_valid_posts(messages)
        logging.info(f"After filtering, {len(valid_messages)} valid messages remain")

        # Find rhyming couplets
        couplets = []
        for i, post1 in enumerate(valid_messages):
            word1 = post1.last_word
            if not word1:
                continue
                
            for post2 in valid_messages[i+1:]:
                word2 = post2.last_word
                if not word2:
                    continue
                    
                # Check if words rhyme and messages are within 24 hours
                if (do_words_rhyme(word1, word2) and 
                    abs((post1.created_at - post2.created_at).total_seconds()) <= 86400):
                    couplets.append(RhymingCouplet(post1, post2))

        return couplets
    
    def find_oldest_recent_couplet(self, couplets: List[RhymingCouplet]) -> Optional[RhymingCouplet]:
        """Find the couplet whose most recent post is the oldest among all couplets."""
        if not couplets:
            return None
            
        return min(couplets, key=lambda x: x.most_recent_timestamp)

if __name__ == "__main__":
    finder = IambicRhymeFinder()
    couplets = finder.find_rhyming_couplets()
    
    print(f"\nFound {len(couplets)} potential rhyming couplets:")
    for couplet in couplets:
        print("\nCouplet:")
        print(f"1: {couplet.first_post.post_text}")
        print(f"2: {couplet.second_post.post_text}")
        print(f"Posted: {couplet.first_post.created_at} and {couplet.second_post.created_at}")
        print(f"Message IDs: {couplet.first_post.message_id} and {couplet.second_post.message_id}")
    
    # Find and print the oldest recent couplet
    oldest_recent_couplet = finder.find_oldest_recent_couplet(couplets)
    if oldest_recent_couplet:
        print("\nCouplet with oldest most recent post:")
        print(f"1: {oldest_recent_couplet.first_post.post_text}")
        print(f"2: {oldest_recent_couplet.second_post.post_text}")
        print(f"Posted: {oldest_recent_couplet.first_post.created_at} and {oldest_recent_couplet.second_post.created_at}")
        print(f"Message IDs: {oldest_recent_couplet.first_post.message_id} and {oldest_recent_couplet.second_post.message_id}")