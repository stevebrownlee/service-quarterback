""" NLP service that processes user queries and stores the results in Redis """
import json
import re
import string
import logging
from collections import Counter
import redis
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("NLPService")

# Initialize the transformer model and Redis connection
transformer_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

def process_queries(queries):
    """ Process the queries and store the results in Redis """
    try:
        # Tokenize and remove stop words
        stop_words = set(stopwords.words('english'))
        punctuation = set(string.punctuation)
        all_stop_words = stop_words.union(punctuation)

        tokens = [word for query in queries for word in word_tokenize(query.lower()) if word not in all_stop_words]
        logger.debug("Generated tokens.")

        # Count frequencies
        word_counts = Counter(tokens)

        # Get the most common words
        common_words = word_counts.most_common(10)
        logger.debug("Found common words.")

        # Find phrases containing these common words
        patterns = []
        seen_sentences = set()
        for word, _ in common_words:
            sentence_pattern = re.compile(r'([^.?!]*\b' + re.escape(word) + r'\b[^.?!]*[.?!])', re.IGNORECASE)
            matches = [sentence_pattern.search(query).group() for query in queries if sentence_pattern.search(query)]
            for match in matches:
                if match not in seen_sentences:
                    seen_sentences.add(match)
                    patterns.append(match)
        logger.debug("Found top sentence patterns.")

        # Generate sentence embeddings
        embeddings = transformer_model.encode(patterns)
        logger.debug("Generated embeddings. Embeddings shape: %s", embeddings.shape)

        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=0.3, min_samples=1, metric='cosine').fit(embeddings)
        logger.debug("Performed DBSCAN clustering.")

        unique_patterns = []
        for cluster_label in set(clustering.labels_):
            if cluster_label != -1:  # Ignore noise points
                cluster_indices = np.where(clustering.labels_ == cluster_label)[0]
                representative_index = cluster_indices[0]
                unique_patterns.append(patterns[representative_index])
        logger.debug("Extracted unique patterns.")

        redis_client.set('search_results', json.dumps(unique_patterns[:10]))
        logger.debug("Stored results in Redis.")

    # pylint: disable=broad-except
    except Exception as e:
        logger.error("Error in process_queries job: %s", e, exc_info=True)

def main():
    """ Main function that listens for messages on the Redis channel """
    pubsub = redis_client.pubsub()
    pubsub.subscribe('channel_help_query')

    logger.info('Waiting for messages. To exit press CTRL+C')
    for message in pubsub.listen():
        if message['type'] == 'message':
            data = json.loads(message['data'])
            queries = data['queries']
            process_queries(queries)

if __name__ == "__main__":
    main()
