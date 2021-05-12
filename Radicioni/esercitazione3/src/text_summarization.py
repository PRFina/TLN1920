import math
from itertools import product
import queue

class TextSummarizer:
    """Wrapper class to summarize with a given compression ratio a textual document using 4 step extractive procedure:
    1. Extract main topic of the document from title.
    2. Extract topic from the document body chunk, ie the actual text content.
    3. Score each segmented chunk of the body (eg. sentence or paragraph) with weight overlap similarity
    4. Reorder scored chunks to sintetize a summarized version of the original text.

    The document body segmentation depends on the actual data_manager.Document instance passed as input    
    to the get_summary() method. This means that this class is unaware of the segmentation level used.

    Here, topic is represented as a collection of Nasari vectors builded from topic_extraction.TopicExtractor
    instances given as input in the object constructor. 
    """

    def __init__(self, main_extractor, chunk_extractor):
       """ Initialize a TextSummarizer instance with given topic extractors.

        Args:
            topic_extractors (dict): dictionary of topic_extraction.TopicExtractor instances (already initialized objects!)
       """
       self._main_topic_extractor = main_extractor
       self._chunk_topic_extractor = chunk_extractor
       self._relevance_queue = queue.PriorityQueue()


    def _summarize(self, document):
        """ Method to build an internal representation of scored body chunks.
        To keep track of score ordering the class use a priority queue.
        The priority queue contains triples:

        (avg. relevance score, chunk order in the original text, textual content of the chunk)

        body_chunk is some textual unit (ie sentence or paragraph), which coarseness level depends on the 
        actual document instance given as input. 
        
        The core operation is the computation of average weighted overlap (WO) relevance 
        between all possible pairs of title vectors and chunk vectors.

        This is an internal method to build the priority queue. To get the actual summarized text
        the "public" version get_summary() must be used.

        Args:
            document (data_manager.Document): parsed textual document to summarize
        """
        # extract title topic
        title_topic = self._main_topic_extractor.get_topic(document)

        for chunk_num, body_chunk in enumerate(document.body, start=1):
            chunk_topic = self._chunk_topic_extractor.get_topic(body_chunk)  # extract chunk topic
            
            # use itertools.product to compute pair-wise cartesian product of nasari vectors between title and chunk
            relevance_scores = [self.weighted_overlap(v1,v2) for v1, v2 in product(title_topic, chunk_topic)]
            
            avg_relevance = sum(relevance_scores) / len(relevance_scores) if len(relevance_scores) > 0 else 0.0

            self._relevance_queue.put((-avg_relevance, chunk_num, body_chunk )) # -avg score since queue is a min heap

    def get_summary(self, document, compression_ratio=10, debug=False):
        """ Get a summarized version of the original document with a given compression ratio.
        Higher the compression ration shorter will be the summary.
        
        
        Args:
            document (data_manager.Document): parsed textual document to summarize
            compression_ratio (int, optional): percentual strength of compression. Defaults to 10.

        Returns:
            str:  
        """
        
        # compute chunks to retain/discard wrt the given compression ratio
        cutoff = int(round((compression_ratio / 100) * len(document.body)))
        to_keep = len(document.body) - cutoff

        self._summarize(document) # build priority queue

        selected_chunks = []
        for i in range(to_keep): # just get a to_keep number of chunks
            selected_chunks.append(self._relevance_queue.get())
        
        self._relevance_queue.queue.clear() # empty the queue for further processing
        
        return self._format_summary(selected_chunks, debug)

    def _format_summary(self, selected_chunks, debug=False):
        """Format selected chunks into a string

        Args:
            selected_chunks (list of triple): list of triples (avg. relevance score, chunk order in the original text, textual content of the chunk)
            debug (bool, optional): build usefull info for debugging purpose. Defaults to False.

        Returns:
            str: formatted textual representation of the summary. 
        """
        format_string = '{} [{:.3f};{}]' if debug else '{}'
        formatted_chunks = [format_string.format(chunk, -relevance, chunk_num) for relevance, chunk_num, chunk in selected_chunks] # -relevance since is inverted in te queue
        
        return '\n'.join(formatted_chunks) # join chunks in multi-line string


    def weighted_overlap(self, v1, v2):
        """ Weighted Overlap (WO) similarity measure by Pilehvar et al. (2013)

        Args:
            v1 (data_manager.Nasari vector): list of tuples (lemma, score)
            v2 (data_manager.Nasari vector): list of tuples (lemma, score)
        """

        # project only lemmas
        v1_components = set([elem[0] for elem in v1]) 
        v2_components = set([elem[0] for elem in v2])

        # components shared by both vectors
        overlapped_components = v1_components.intersection(v2_components) 
        
        wo = 0
        if len(overlapped_components) > 0:
            num = sum([1.0/(self._rank(q,v1) + self._rank(q,v2)) for q in overlapped_components])
            den = sum([1.0/(2*(i+1)) for i,_ in enumerate(overlapped_components)])
            wo = num / den
        
        return math.sqrt(wo) # squared root as suggested by Navigli et al.

    
    def _rank(self, component, vector):
        """ Compute the rank of a nasari vector component, ie the position of the component (lemma) 
        in the ranking created by descending scores values"
        Args:
            component (str): [description]
            vector (data_manager.Nasari vector): list of tuples (lemma, score)

        Returns:
            int: rank of the component
        """

        # make sure that lemmas as ordered by score
        ranking = sorted(vector, key=lambda v: v[1], reverse=True) # descending order by score
        
        rank_pos= 0
        for pos, (lemma, score) in enumerate(vector, start=1):
            if lemma == component:
                rank_pos= pos
                break
        
        return rank_pos