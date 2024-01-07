# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import spacy
import nltk
import spacy.cli

def loadData():
    """
    Downloads necessary NLTK and SpaCy resources and loads text data from the 'milton-paradise.txt' file in the Gutenberg corpus.
    
    Returns:
    chars (str): Raw text content.
    words (list): List of words in the text.
    sents (list): List of sentences in the text.
    """
    nltk.download("gutenberg")
    nltk.download("punkt")
    # Download the en_core_web_sm model
    spacy.cli.download("en_core_web_sm")
    
    #LOADING DATA
    ################################################################################
    chars = nltk.corpus.gutenberg.raw('milton-paradise.txt')
    words = nltk.corpus.gutenberg.words('milton-paradise.txt')
    sents = nltk.corpus.gutenberg.sents('milton-paradise.txt')
    ################################################################################
    return chars,words,sents
    
def statistics(chars, words, sents):
    """
    Computes various statistical measures based on the input text data.

    Parameters:
    chars (str): Raw text content.
    words (list): List of words in the text.
    sents (list): List of sentences in the text.

    Returns:
    word_per_sent (int): Average number of words per sentence.
    char_per_word (int): Average number of characters per word.
    char_per_sent (int): Average number of characters per sentence.
    longest_sentence (int): Length of the longest sentence in characters.
    longest_word (int): Length of the longest word in characters.
    l_sentw (int): Length of the longest sentence in words.
    """
    
    word_lens = []
    for word in words:
        word_lens.append(len(word))
    
    sent_lens = []
    for sentence in sents:
        sent_lens.append(len(sentence))
    #print(sents[0])
    
    #print(len(sents[1]), sents[1])
    chars_in_sents =[]
    for characters in sents:
        chars_in_sents.append(len("".join(characters)))
    #print(chars_in_sents)
    
    word_per_sent = round(sum(sent_lens) / len(sents))
    char_per_word = round(sum(word_lens) / len(words))
    char_per_sent = round(sum(chars_in_sents) / len(sents))
    
    longest_sentence = max(chars_in_sents)
    l_sentw = max(sent_lens)
    longest_word = max(word_lens)
    
    return word_per_sent, char_per_word, char_per_sent, longest_sentence, longest_word, l_sentw
    
def nbest(d, n=1):
    """
    Retrieves the top n frequent words

    Parameters:
    d (dict): Input dictionary with string keys and numeric values.
    n (int): Number of top values to retrieve.

    Returns:
    dict: Dictionary containing the top n key-value pairs.
    """
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=True)[:n])

def spacytknz(chars):
    """
    Tokenizes the input text using SpaCy.

    Parameters:
    chars (str): Raw text content.

    Returns:
    wordsS (list): List of words in the text processed by SpaCy.
    sentsS (list): List of sentences in the text processed by SpaCy.
    """
    nlp = spacy.load("en_core_web_sm")
    wordsS = nlp(chars)
    sentsS = list(wordsS.sents)
    wordsS = [token.text for token in wordsS]
    sentsS = [sent for sent in sentsS]
    aux = []
    for i in range(0,len(sentsS)):
        for word in sentsS[i]:
            aux.append(word.text)
        sentsS[i] = aux
        aux = []
    return wordsS,sentsS

def nltktknz(chars):
    """
    Tokenizes the input text using NLTK.

    Parameters:
    chars (str): Raw text content.

    Returns:
    wordsN (list): List of words in the text processed by NLTK.
    sentsN (list): List of sentences in the text processed by NLTK.
    """
    sentsN = nltk.sent_tokenize(chars)
    wordsN = nltk.word_tokenize(chars)
    
    for i in range(0,len(sentsN)):
        sentsN[i] = nltk.word_tokenize(sentsN[i])
        
    return wordsN,sentsN
        
def printStat(word_per_sent, char_per_word, char_per_sent, longest_sent, longest_word, l_sentw):
    """
    Prints statistical measures.

    Parameters:
    word_per_sent (int): Average number of words per sentence.
    char_per_word (int): Average number of characters per word.
    char_per_sent (int): Average number of characters per sentence.
    longest_sent (int): Length of the longest sentence in characters.
    longest_word (int): Length of the longest word in characters.
    l_sentw (int): Length of the longest sentence in words.
    """
    print('Word per sentence', word_per_sent)
    print('Char per word', char_per_word)
    print('Char per sentence', char_per_sent)
    print('Longest sentence (characters)', longest_sent)
    print('Longest sentence (words)', l_sentw)
    print('Longest word', longest_word)
    
def topFreq(words):
    """
    Computes and prints the top frequency statistics of a list of words.

    Parameters:
    words (list): List of words.

    Prints:
    Upper case Lexicon size, Top 5 frequencies in upper case.
    Lower case Lexicon size, Top 5 frequencies in lower case.
    """
    #calculate the frequencies with NLKT
    freq_dist = nltk.FreqDist(words)
    #get the n most frequent tokens
    print('Upper case:', len(freq_dist), 'Top 5:', nbest(freq_dist, n=5)) 
    
    #put in lower caps
    lexicon = ([w.lower() for w in words])
    #Calculate freq using counter
    l_freq_dist = nltk.FreqDist(lexicon)
    
    #get the n most frequent tokens
    print("Lower case:", len(l_freq_dist), 'Top 5:', nbest(l_freq_dist, n=5)) 