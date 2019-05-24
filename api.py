from flask import Flask
from flask_restful import Resource, Api
from flask_cors import CORS
from flask import render_template, make_response
from LyricsAnalyzer import LyricsAnalyzer
import nltk
app = Flask(__name__)
api = Api(app)
CORS(app)


def encode_mostCommon(mostCommonWords):
    """
    Encode a most common words results to json
    Input: [("word1",count1),("word2",count2),...,("wordn",countn),]
    Output:{"mostCommon": [{"word": word1, "count": count1},..]}
    """
    wordList = []
    for pair in mostCommonWords:
        word, count = pair
        wordList.append({'word': word, 'count': count})
    
    return {'mostCommon': wordList}

class WordTokensBase(Resource):
    def get(self,lyrics):
        analyzer = LyricsAnalyzer(lyrics)
        return {'wordTokens': analyzer.wordTokenize()}

class WordTokensNoStops(Resource):
    def get(self,lyrics):
        analyzer = LyricsAnalyzer(lyrics)
        return {'wordTokens': analyzer.wordTokenizeWithoutStopwords()}

class MostCommonBase(Resource):
    def get(self,lyrics, num):
        analyzer = LyricsAnalyzer(lyrics)
        
        #Get the most common words from the analyzer, and encode them to json
        return encode_mostCommon(analyzer.mostCommon(num))

class MostCommonBaseDefault(Resource):
    def get(self,lyrics):
        analyzer = LyricsAnalyzer(lyrics)
        #Get the most common words from the analyzer, and encode them to json
        return encode_mostCommon(analyzer.mostCommon(10))

class MostCommonFiltered(Resource):
    def get(self,lyrics, num):
        analyzer = LyricsAnalyzer(lyrics)
        return encode_mostCommon(analyzer.mostCommonFiltered(num))

class MostCommonFilteredDefault(Resource):
    def get(self,lyrics):
        analyzer = LyricsAnalyzer(lyrics)
        return encode_mostCommon(analyzer.mostCommonFiltered(10))

class GetSentimentVADER(Resource):
    def get(self,lyrics):
        return LyricsAnalyzer(lyrics).getSentiment()

class GetAnalysisDefault(Resource):
    def get(self,lyrics):
        analyzer = LyricsAnalyzer(lyrics)
        mostCommon, sentiment = analyzer.getAnalysis(10)

        return {'words':encode_mostCommon(mostCommon), 'sentiment': sentiment}

class GetAnalysis(Resource):
    def get(self,lyrics,num):
        analyzer = LyricsAnalyzer(lyrics)
        mostCommon, sentiment = analyzer.getAnalysis(num)

        return {'words':encode_mostCommon(mostCommon), 'sentiment': sentiment}

class Documentation(Resource):
    def get(self):
        headers = {'Content-Type': 'text/html'}
        return make_response(render_template('documentation.html'),200,headers)

#Tokenizing a text into words
api.add_resource(WordTokensBase,'/tokens/base/<string:lyrics>')
api.add_resource(WordTokensNoStops,'/tokens/noStopwords/<string:lyrics>')

#Find the most common words in the text
api.add_resource(MostCommonBase,'/mostCommon/base/<string:lyrics>/<int:num>')
api.add_resource(MostCommonBaseDefault,'/mostCommon/base/<string:lyrics>')
api.add_resource(MostCommonFiltered,'/mostCommon/filtered/<string:lyrics>/<int:num>')
api.add_resource(MostCommonFilteredDefault,'/mostCommon/filtered/<string:lyrics>')

#Find the sentiment of a text, using the VADER algorithm:
api.add_resource(GetSentimentVADER,'/sentiment/vader/<string:lyrics>')

#Do an analysis of a text, providing most common words and sentiment
api.add_resource(GetAnalysisDefault,'/analysis/<string:lyrics>')
api.add_resource(GetAnalysis,'/analysis/<string:lyrics>/<int:num>')

#Documentation
api.add_resource(Documentation, '/documentation')


if __name__ == '__main__':
    nltk.download('vader_lexicon')
    nltk.download('popular')
    app.run(host='0.0.0.0', port=80)
