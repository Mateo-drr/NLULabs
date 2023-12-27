# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import spacy
import nltk
import spacy.cli

def loadData():
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
    get n max values from a dict
    :param d: input dict (values are numbers, keys are stings)
    :param n: number of values to get (int)
    :return: dict of top n key-value pairs
    """
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=True)[:n])