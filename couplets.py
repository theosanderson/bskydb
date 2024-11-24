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
import time

import re
from collections import defaultdict
from itertools import product
from functools import lru_cache

# Load NLTK's CMU Pronouncing Dictionary
pronouncing_dict = cmudict.dict()
# overwrite "a" to be pronounced as "ah" instead of "ey"
pronouncing_dict["a"] = [["AH0"]]
pronouncing_dict["perfect"] = [["P", "ER1", "F", "IH0", "K", "T"]]

# Precompute stress patterns for each word
word_stress_patterns = defaultdict(list)
for word, pronunciations in pronouncing_dict.items():
    word_stress_patterns[word] = [
        [int(phoneme[-1]) for phoneme in pron if phoneme[-1].isdigit()]
        for pron in pronunciations
    ]

# Helper to convert numbers to words
def num2words(num):
    under_20 = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen',
                'eighteen', 'nineteen']
    tens = ['twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
    above_100 = {100: 'hundred', 1000: 'thousand'}

    if num < 20:
        return under_20[num]
    if num < 100:
        return tens[(num // 10) - 2] + ('' if num % 10 == 0 else ' ' + under_20[num % 10])
    pivot = max([key for key in above_100.keys() if key <= num])
    return num2words(num // pivot) + ' ' + above_100[pivot] + ('' if num % pivot == 0 else ' ' + num2words(num % pivot))

# Normalize text
def normalize_text(text):
    substitutions = {'&': ' and ', 'w/': 'with', 'w/o': 'without'}
    for k, v in substitutions.items():
        text = text.replace(k, v)
    text = re.sub(r'\d+', lambda m: num2words(int(m.group())), text)
    text = re.sub(r'[^a-zA-Z\s\']', '', text)
    return re.sub(r'\s+', ' ', text.strip().lower())

# Check if a stress pattern conforms to iambic pentameter
def is_iambic_pentameter(pattern):
    return len(pattern) == 10 and all(
        (stress == 0) if i % 2 == 0 else (stress == 1)
        for i, stress in enumerate(pattern)
    )

# Main function to check if text is in iambic pentameter
@lru_cache(maxsize=1000)
def check_iambic_pentameter(raw_text):
    if '\n' in raw_text:
        print("Not iambic as contained linebreak")
        return False
    text = normalize_text(raw_text)
    words = text.split()


    
    if not words or any(word not in word_stress_patterns for word in words):
        print("Not iambic due to unexpected word")
        return False
    
    stress_combinations = [word_stress_patterns[word] for word in words]
    all_stress = product(*stress_combinations)

    
    found_iambic = False
    patterns_checked = 0
    for combination in all_stress:
        pattern = [stress for stresses in combination for stress in stresses]
        patterns_checked += 1
        
        if is_iambic_pentameter(pattern):
            found_iambic = True
        

    # if checked too many patterns, return False
    if patterns_checked > 24:
        print("not iambic as too many patterns")
        return False
    return found_iambic




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

def get_all_phoneme_endings(word):
    """Get all possible ending phoneme sequences for a word."""
    if word.lower() not in pronouncing_dict:
        return []
    endings = []
    for pronunciation in pronouncing_dict[word.lower()]:
        for i, phoneme in enumerate(reversed(pronunciation)):
            if any(char.isdigit() for char in phoneme):
                endings.append(tuple(pronunciation[-i-1:]))
                break
        else:
            endings.append(tuple(pronunciation))
    return endings

def do_words_rhyme(word1, word2):
    """Check if two words rhyme, considering all possible pronunciations."""
    if word1.lower() == word2.lower():
        return False
    
    phoneme_endings1 = get_all_phoneme_endings(word1)
    phoneme_endings2 = get_all_phoneme_endings(word2)
    
    
    if not phoneme_endings1 or not phoneme_endings2:
        print("no phoneme endings for", word1, word2)
        return False
    
    # if either has multiple possible endings, then it is ambiguous and we should return False
    if len(phoneme_endings1) > 1 or len(phoneme_endings2) > 1:
        print("ambiguous endings for", word1, word2)
        return False
    # if the endings are the same, then they rhyme
    if phoneme_endings1[0] == phoneme_endings2[0]:
        return True

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

    def update_repost_timestamp(self, post: BlueskyPost):
        """Update the repost timestamp for a post in the database."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                query = """
                    UPDATE iambic_messages
                    SET reposted_at = NOW()
                    WHERE id = %s
                """

                cur.execute(query, (post.id,))
                conn.commit()
                print(f"Updated repost timestamp for post {post.uri}")
                
    def repost_post(self, post: BlueskyPost):
        """Repost a single post and update its timestamp."""
        try:
            # Get post details from API to get CID
            post_details = self.client.app.bsky.feed.get_posts({'uris': [post.uri]}).posts[0]
            # Repost using simplified client method
            self.client.repost(uri=post.uri, cid=post_details.cid)
            print(f"Reposting post: {post.uri}")
            self.update_repost_timestamp(post)
            logging.info(f"Successfully reposted post: {post.uri}")
        except Exception as e:
            logging.error(f"Failed to repost {post.uri}: {e}")
            raise

    def repost_couplet(self, couplet: RhymingCouplet):
        """Repost both posts in chronological order."""
        # Get posts in chronological order
        posts = sorted([couplet.first_post, couplet.second_post], 
                        key=lambda x: x.created_at)
        
        # Repost older post first
        self.repost_post(posts[0])
        time.sleep(2)  # Brief pause between reposts
        # Repost newer post
        self.repost_post(posts[1])

    def get_new_messages(self, last_timestamp: datetime) -> Iterator[BlueskyPost]:
        """Get all messages since the last reposted timestamp as BlueskyPost objects."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                
                query = """
                    SELECT id, message_id, post_text, created_at, did
                    FROM iambic_messages
                    WHERE created_at > %s
                    AND created_at < NOW() - INTERVAL '1 hour'
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
        - Not a reply
        """
        try:
            logging.info(post.indexed_at)
            logging.info(post.record.created_at)
            # if created at at indexed at are more than 1 day apart in either direction, skip
            logging.info(post.record)
            #indexed_at_unix_timestamp = datetime.strptime(post.indexed_at, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()
            #created_at_unix_timestamp = datetime.strptime(post.record.created_at, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()
            
            
            # if created before 2024, reject
            # year is first 4 chars of created_at
            year = post.record.created_at[0:4]
            if int(year)<2024:
                print("discarding as year not 2024")
                return False
            
            if post.embed is not None:
                print("disarding as had embed")
                return False
            
    
            if post.record.reply:
                print("Reply")
                return False
            print("Not a reply")
        
            return True
        except Exception as e:
            logging.error(f"Error validating post: {e}")
            return False

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

        # Filter out messages that are not in iambic pentameter, logging those excluded
        old_messages = messages
        messages = []
        for message in old_messages:
            if check_iambic_pentameter(message.post_text):
                messages.append(message)
            else:
                logging.info(f"Filtered out message: {message.post_text}")


        # Fetch and validate posts
        valid_messages = self.fetch_valid_posts(messages)
        logging.info(f"After filtering, {len(valid_messages)} valid messages remain")
        for message in valid_messages:
            logging.info(f"Valid message: {message.post_text}")
           # logging.info(f"{message.record.created_at}")

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
                    
                if (do_words_rhyme(word1, word2) ):
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
        # raise Exception("Test")
        # Repost the couplet in chronological order
        try:
            finder.repost_couplet(oldest_recent_couplet)
            print("Successfully reposted couplet in chronological order!")
        except Exception as e:
            print(f"Failed to repost couplet: {e}")