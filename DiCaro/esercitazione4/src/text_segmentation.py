import scipy.spatial.distance as distance
import itertools as it

def cosine_similarity(v1,v2):
    return 1 - distance.cosine(v1,v2)

def preprocessor(chunk):
    bow = set(nltk.word_tokenize(chunk.lower()))
    stop_words = set(stopwords.words('english')) 
    return bow.difference(stop_words)


class TextSegmenter:

    def __init__(self, preprocessor, embedding, similarity_func) -> None:
        self._preprocessor = preprocessor
        self._embedding = embedding
        self._similarity_func = similarity_func


    def segment(self, document, initial_blocks=25, smooth=False, window_size=2):
        # split chunks into initial_blocks number of blocks
        blocks = np.array_split(document.get_chunks(), initial_blocks)
        sims=[]

        # loop trough adjacents blocks, starting from the first one
        for current_block, next_block in zip(blocks[::],blocks[1::]):

            block_sim = self.block_similarity(current_block,next_block)
            sims.append(block_sim)

        # find local minima valleys
        valleys, block_similarities = self.find_valleys(np.array(sims), smooth, window_size)

        return valleys, block_similarities, blocks


    def block_similarity(self, block1, block2):

        # each block is a list of consecutive chunks (eg. sentences, paragraphs, or coarser informative units)
        # each chunk is a list of consecutive tokens, the chunk tokenization logic is delegated to preprocessor  

        chunk_sims = []
        for (chunk1, chunk2) in it.product(block1, block2):
            bow1 = self._preprocessor(chunk1)
            bow2 = self._preprocessor(chunk2)

            token_sims = []
            for (token1, token2) in it.product(bow1, bow2):
                v1 = self._embedding[token1]
                v2 = self._embedding[token2]

                if v1 is not None and v2 is not None:
                    token_sim = self._similarity_func(v1,v2)
                else:
                    token_sim =  0 
                
                token_sims.append(token_sim)

            avg_chunk_sim = np.array(token_sims).mean()
            chunk_sims.append(avg_chunk_sim)

        return np.array(chunk_sims).mean() # avg blocks sim

    
    def find_valleys(self, similarities, smooth=False, window_size=2):
        def moving_average(x, w):
            return np.convolve(x, np.ones(w), 'valid') / w
        
        # smooth the similarity signal
        if smooth:
            similarities = moving_average(similarities, window_size)
        
        # find local minima
        peaks, _ = signal.find_peaks(-similarities) # invert to find negative peaks
        
        return peaks, similarities