import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, TweetTokenizer
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment import util
from nltk.sentiment import SentimentIntensityAnalyzer
"""
Internet guide that was used:
https://www.datacamp.com/community/tutorials/text-analytics-beginners-nltk
"""

class LyricsAnalyzer():
    """
    Class that can perform Natural Language Processing NLP on a given text string.
    """
    ##lyrics = ""

    def __init__(self, lyrics):
        
        #Remove unnecessary artefacts:
        
        #Remove backslashes
        self.lyrics = lyrics.replace("\\","")
        #Remove commas, exclamation points
        self.lyrics = self.lyrics.translate(str.maketrans(',?!',"   "))

        #If any html breaks are present, they should be removed too
        if "<br>" in self.lyrics:
            self.removeHtmlBreaks()

    def sentenceTokenize(self):
        """
        Tokenize the text, breaking it up into sentence tokens
        """
        return sent_tokenize(self.lyrics)

    def wordTokenize(self):
        """
        Tokenize the text, breaking it up into word tokens
        In this tokenizer, "couldn't" and other terms like it
        art treated as a single word
        """
        tokenizer = TweetTokenizer()
        return tokenizer.tokenize(self.lyrics)

    def toString(self):
        return self.lyrics

    def mostCommon(self, n):
        return FreqDist(self.wordTokenize()).most_common(n)

    def mostCommonFiltered(self, n):
        """
        Filtered out stopwords, all words are stemmed
        """

        tokensNoStopwords = self.wordTokenizeWithoutStopwords()
        wordsStemmed = []
        ps = PorterStemmer()

        for word in tokensNoStopwords:
            wordsStemmed.append(ps.stem(word))
        

        return FreqDist(wordsStemmed).most_common(n)

    def wordTokenizeWithoutStopwords(self):
        tokens = self.wordTokenize()
        result = []
        stopWordsIterable = set(stopwords.words("english"))
        for token in tokens:
            if token not in stopWordsIterable:
                result.append(token)
        return result

    def getSentiment(self):

        vader_analyzer = SentimentIntensityAnalyzer()
        return vader_analyzer.polarity_scores(self.lyrics)

    def removeHtmlBreaks(self):
        self.lyrics = self.lyrics.replace("<br>"," ")
    
    def getAnalysis(self,n):
        mostCommon = self.mostCommon(n)
        polarity_scores = self.getSentiment()
        return mostCommon,polarity_scores
