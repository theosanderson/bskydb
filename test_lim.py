import psycopg2
from datetime import datetime
import nltk
from nltk.corpus import cmudict
import itertools
import re
from tqdm import tqdm
from collections import defaultdict
from functools import lru_cache

# Load and cache the CMU dict
nltk.download('cmudict', quiet=True)
pronouncing_dict = cmudict.dict()

# Create a cached version of word stress patterns
word_stress_patterns = defaultdict(list)
for word, pronunciations in pronouncing_dict.items():
    word_stress_patterns[word] = [
        [int(phoneme[-1]) for phoneme in pron if phoneme[-1] in '012']
        for pron in pronunciations
    ]

@lru_cache(maxsize=10000)
def get_rhyme_sound(word):
    """Get the rhyme sound of a word (last stressed vowel and following sounds)."""
    word = word.lower()
    if word not in pronouncing_dict:
        return None
    
    pron = pronouncing_dict[word][0]  # Use first pronunciation
    try:
        # Find the last stressed vowel
        for i in range(len(pron) - 1, -1, -1):
            if any(char.isdigit() for char in pron[i]) and '1' in pron[i]:
                return tuple(pron[i:])  # Return from stressed vowel to end
    except:
        return None
    return None

def do_lines_rhyme(line1, line2):
    """Check if two lines rhyme based on their last words."""
    # Get last word of each line

    words1 = re.sub(r'[^A-Za-z\s]', '', line1.lower()).split()
    words2 = re.sub(r'[^A-Za-z\s]', '', line2.lower()).split()
    
    if not words1 or not words2:
        return False
    
    last_word1 = words1[-1]
    last_word2 = words2[-1]
    
    # Don't count identical words as rhyming
    if last_word1 == last_word2:
        return False
    
    # Get rhyme sounds
    sound1 = get_rhyme_sound(last_word1)
    sound2 = get_rhyme_sound(last_word2)
    
    return sound1 is not None and sound2 is not None and sound1 == sound2

def is_iambic_pentameter(pattern):
    """Check if a stress pattern is iambic pentameter."""
    return len(pattern) == 10 and all(
        (s == 0 or s == 2) if i % 2 == 0 else s == 1
        for i, s in enumerate(pattern)
    )

def is_anapestic_trimeter(pattern):
    """
    Check if a stress pattern is anapestic trimeter, allowing for one mismatch.
    Anapestic trimeter should follow the pattern: 112112112 (where 2 is stressed).
    Returns True if the pattern matches with at most one deviation.
    
    Args:
        pattern: List of integers where 1 represents unstressed and 2 represents stressed syllables
        
    Returns:
        bool: True if pattern is anapestic trimeter (allowing one mismatch), False otherwise
    """
   
        
    def get_expected_stress(position):
        """Return expected stress level (1 or 2) for given position in anapestic trimeter."""
        return 2 if position % 3 == 0 else 1
    
    mismatches = sum(
        actual != get_expected_stress(i)
        for i, actual in enumerate(pattern)
    )
    
    return mismatches <= 1

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
    # find the appropriate pivot
    pivot = max([key for key in above_100.keys() if key <= num])
    return num2words(num // pivot) + ' ' + above_100[pivot] + ('' if num % pivot == 0 else ' ' + num2words(num % pivot))


def numerals_to_words(text):
    """Convert numerals to words in a string
    E.g. I have 2 apple and 23 oranges -> I have two apple and twenty three oranges

    Args:
        text (str): Text to convert
    """
    text = re.sub(r'(\d+)', lambda m: num2words(int(m.group(0))), text)
    return text
    

@lru_cache(maxsize=10000)
def check_iambic_pentameter(text):
    text = "Who carried his food in a bucket"
    """Optimized check for iambic pentameter."""
    # Quick length check before doing more work
    if len(text.split()) < 5 or len(text.split()) > 75:
        return False
    
    # Check for banned characters
    if any(b in text for b in banned):
        return False
    
    # Substitute banned characters
    for k, v in substitutions.items():
        text = text.replace(k, v)
    
    # Clean text
    text = numerals_to_words(text)
    text = re.sub(r'[^A-Za-z\s\']', '', text.lower())
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    
    if len(set(words)) == 1:
        return False
    
    if not all(word in word_stress_patterns for word in words):
        return False
    
    stress_options = [word_stress_patterns[word] for word in words]
    #stress_options = [[s[0]] for s in stress_options]
    #raise Exception(stress_options)
    
    if sum(len(p) > 1 for p in stress_options) > 3:
        return False    
    
    min_syllables = sum(min(len(patterns) for patterns in p) for p in stress_options)
    max_syllables = sum(max(len(patterns) for patterns in p) for p in stress_options)
    if min_syllables > 10 or max_syllables < 6:
        return False
    
    for stress_combination in itertools.product(*stress_options):
        pattern = [s for stresses in stress_combination for s in stresses]
        if is_anapestic_trimeter(pattern):
            print (pattern)
            print (text)
            return True
    return False

# Database connection parameters
db_params = {
    'dbname': 'bluesky_posts',
    'user': 'postgres',
    'password': 'dev-password-123',
    'host': 'localhost',
    'port': '5432'
}

BATCH_SIZE = 50000

try:
    # Connect to the database
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()
    
    # Get total count of posts
    cur.execute("SELECT COUNT(*) FROM posts")
    total_posts = cur.fetchone()[0]
    
    print(f"Total posts to analyze: {total_posts}")
    iambic_posts = []  # Store all iambic pentameter posts
    
    # Process in batches
    for offset in tqdm(range(0, total_posts, BATCH_SIZE)):
        cur.execute("""
            SELECT message_id, did, post_text, timestamp_us, created_at 
            FROM posts 
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """, (BATCH_SIZE, offset))
        
        batch = cur.fetchall()
        
        for post in batch:
            message_id, did, text, timestamp_us, created_at = post
            
            if not text:
                continue
                
            if check_iambic_pentameter(text):
                post_time = datetime.fromtimestamp(timestamp_us / 1_000_000)
                iambic_posts.append({
                    'message_id': message_id,
                    'did': did,
                    'text': text,
                    'timestamp': post_time
                })
    
    print(f"\nFound {len(iambic_posts)} posts in iambic pentameter")

    # Print them
    for post in iambic_posts:
        
        print(f"{post['text']}")
        
    print("\nSearching for rhyming couplets...")
    
    # Find rhyming couplets
    rhyming_pairs = []
    for i in range(len(iambic_posts)):
        for j in range(i + 1, len(iambic_posts)):
            if do_lines_rhyme(iambic_posts[i]['text'], iambic_posts[j]['text']):
                rhyming_pairs.append((iambic_posts[i], iambic_posts[j]))

    
    
    # Print rhyming couplets
    print(f"\nFound {len(rhyming_pairs)} rhyming couplets!")
    for pair in rhyming_pairs:
        print("\n" + "=" * 80)
        print(f"Couplet found:")
        print(f"1: {pair[0]['text']}")
        print(f"2: {pair[1]['text']}")
        print(f"Posted by: {pair[0]['did']} at {pair[0]['timestamp']}")
        print(f"         & {pair[1]['did']} at {pair[1]['timestamp']}")
        print("=" * 80)
    
except psycopg2.Error as e:
    print(f"Database error: {e}")

except Exception as e:
    print(f"Error: {e}")

finally:
    if 'cur' in locals():
        cur.close()
    if 'conn' in locals():
        conn.close()