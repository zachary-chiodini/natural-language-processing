import random, re
from math import log
from typing import Dict, List, Set, Tuple

class NaiveBayes :

    Count, Probaility = int, float
    Class, Output, Document, Word = str, str, str, str

    def __init__( self ) -> None :
        self.logPc  : Dict[ Class, Probability ] = {} 
        self.logPwc : Dict[ Class, Dict[ Word, Probability ] ] = {}
        self.result : List[ Tuple[ Class, Output ] ] = []
        self.vocabulary : Dict[ Class, Set[ Document ] ] = {}
        self.classvocab : Dict[ Class, Set[ Document ] ] = {}
        self.accuracy   : float = 0.0
        self.stop_words = {
            'if', 'might', 'big', 'opens', 'but', 'got',
            'almost', 'differently', 'since', 'why', 'things',
            'under', 'perhaps', 'grouped', 'whose', 'show',
            'say', 'first', 'us', 'used', 'room', 'that',
            'seems', 'groups', 'over', 'we', 'whether',
            'wants', 'thus', 'number', 'four', 'you',
            'anywhere', 'smaller', 'within', 'man', 'already',
            'may', 'second', 'though', 'to', 'furthering',
            'finds', 'across', 'through', 'give', 'needing',
            'turns', 'see', 'really', 'interesting', 'what',
            'while', 'ever', 'yet', 'latest', 'greater', 'be',
            'forth', 'nowhere', 'end', 'is', 'on', 'everyone',
            'clear', 'where', 'all', 'area', 'anybody', 'turn',
            'made', 'your', 'ten', 'place', 'faces', 'our',
            'here', 'clearly', 'than', 'something', 'downed',
            'sure', 'asks', 'backs', 'seven', 'everywhere',
            'parts', 'went', 'herself', 'youngest', 'important',
            'put', 'uses', 'around', 'until', 'during', 'these',
            'younger', 'just', 'whole', 'come', 'became',
            'large', 'off', 'mrs', 'seventh', 'must', 'being',
            'presenting', 'rooms', 'his', 'toward', 'into',
            'early', 'each', 'possible', 'later', 'everybody',
            'open', 'highest', 'was', 'more', 'interested',
            'backed', 'several', 'showing', 'most', 'interests',
            'year', 'certain', 'well', 'fully', 'have',
            'wanting', 'last', 'double', 'against', 'down',
            'high', 'had', 'still', 'far', 'point', 'ordering',
            'began', 'good', 'puts', 'then', 'says', 'opening',
            'any', 'evenly', 'given', 'long', 'who', 'because',
            'again', 'away', 'others', 'downs', 'and', 'largely',
            'three', 'nine', 'worked', 'higher', 'let', 'both',
            'right', 'shows', 'has', 'taken', 'become', 'sees',
            'men', 'my', 'making', 'problem', 'back', 'been',
            'how', 'did', 'among', 'it', 'mostly', 'per',
            'places', 'turning', 'saw', 'sides', 'himself',
            'out', 'part', 'cases', 'opened', 'ending',
            'although', 'seeming', 'member', 'therefore',
            'pointed', 'fact', 'older', 'took', 'ended', 'two',
            'other', 'above', 'once', 'keeps', 'think', 'eighth',
            'there', 'use', 'enough', 'does', 'few', 'every',
            'longest', 'somebody', 'way', 'present', 'needed',
            'new', 'do', 'felt', 'side', 'he', 'interest', 'make',
            'such', 'even', 'nothing', 'knows', 'needs', 'goods',
            'itself', 'final', 'working', 'best', 'much',
            'points', 'when', 'of', 'she', 'no', 'respectively',
            'which', 'five', 'yours', 'thing', 'should', 'many',
            'full', 'next', 'twice', 'oldest', 'longer', 'for',
            'without', 'upon', 'seem', 'anything', 'backing',
            'in', 'they', 'lets', 'so', 'smallest', 'came',
            'cannot', 'said', 'someone', 'general', 'showed',
            'from', 'less', 'downing', 'noone', 'tenth', 'sixth',
            'case', 'wanted', 'works', 'thoughts', 'numbers',
            'need', 'different', 'find', 'shall', 'knew', 'parting',
            'wells', 'facts', 'together', 'him', 'myself', 'gets',
            'ways', 'least', 'having', 'one', 'after', 'areas',
            'take', 'ends', 'get', 'this', 'either', 'members',
            'small', 'kind', 'six', 'great', 'some', 'would',
            'quite', 'however', 'like', 'years', 'state', 'or',
            'face', 'also', 'furthers', 'non', 'behind', 'group',
            'know', 'with', 'them', 'necessary', 'order', 'rather',
            'will', 'along', 'by', 'generally', 'gave', 'thought',
            'presents', 'further', 'grouping', 'greatest', 'old',
            'between', 'nobody', 'third', 'mr', 'those', 'eight',
            'thinks', 'work', 'alone', 'furthered', 'their', 'the',
            'seemed', 'her', 'newer', 'ninth', 'can', 'about',
            'before', 'only', 'go', 'likely', 'not', 'gives',
            'presented', 'another', 'at', 'parted', 'newest', 'very',
            'ask', 'asking', 'turned', 'states', 'fifth', 'beings',
            'up', 'often', 'same', 'known', 'problems', 'differ',
            'somewhere', 'keep', 'certainly', 'anyone', 'too',
            'going', 'want', 'me', 'seconds', 'never', 'triple',
            'young', 'its', 'as', 'everything', 'were', 'asked',
            'done', 'pointing', 'ordered', 'an', 'orders', 'could',
            'better', 'are', 'becomes', 'today', 'always', 'now'
            }
        return

    def train(
        self : object,
        data : Dict[ Class, Set[ Document ] ]
        ) -> None :
        '''
        Naive Bayes Training Algorithm
        '''
        self.vocabulary = self.__extract( data )
        self.vocabulary = self.__removeFirstQ( self.vocabulary )
        self.classvocab = self.__classVocab( data, self.vocabulary )
        for label in data :
            self.logPwc[ label ] = {}
            self.logPc[ label ]  = log( len( data[ label ] ) / len( data ) )
            total_count = sum( self.classvocab[ label ].values() )
            for word in self.vocabulary :
                count = self.classvocab[ label ][ word ]
                self.logPwc[ label ][ word ] = log(
                    ( count + 1 ) / ( total_count + len( self.vocabulary ) )
                    )
        return

    def test(
        self : object,
        data : Dict[ Class, Set[ Document ] ]
        ) -> None :
        '''
        Naive Bayes Testing Algorithm
        '''
        assert self.logPwc, 'A Naive Bayes model must be trained.'
        tot_docs = 0
        self.result = []
        self.accuracy = 0.0
        for label in data :
            for document in data[ label ] :
                tot_docs += 1
                output = self.output( document )
                self.result.append( ( label, output ) )
                self.accuracy += ( label == output )
        self.accuracy = self.accuracy / tot_docs if tot_docs else 0
        return

    def trainAndTest(
        self : object,
        data : Dict[ Class, Set[ Document ] ],
        ratio : float = 0.5 # ratio of data to train model
        ) -> None :
        '''
        Train and Test the Naive Bayes Model
        '''
        test  : Dict[ Class, List[ Document ] ] = {}
        train : Dict[ Class, List[ Document ] ] = {}
        for label in data :
            index = int( ratio * len( data[ label ] ) )
            shuffle = list( data[ label ] )
            random.shuffle( shuffle )
            train[ label ] = shuffle[ : index ]
            test[ label ] = shuffle[ index : ]
        self.train( train )
        self.test( test )
        print( 'Examples in Training Set:',
               sum( len( train[ label ] ) for label in train ) )
        print( 'Examples Tested         :',
               sum( len( test[ label ] ) for label in test ) )
        print( 'Total Examples          :',
               sum( len( data[ label ] ) for label in data ) )
        if self.result :
            print( 'Model Accuracy          : {}%'.format(
                round( 100*self.accuracy, 2 ) ) )
        else :
            print( 'Model Accuracy          : Unknown' )
        return

    def output( self : object, document : str ) -> Class :
        '''
        Output of Naive Bayes Model with Document Input
        '''
        assert self.logPwc, 'A Naive Bayes model must be trained.'
        result : Dict[ Class, Probability ] = {}
        for label in self.classvocab :
            result[ label ] = self.logPc[ label ]
            for word in re.findall(
                pattern = '\\b[a-z]{2,}\\b',
                string  = document
                ) :
                if word in self.vocabulary :
                    result[ label ] += self.logPwc[ label ][ word ]
        label, prob = zip( *result.items() )
        return label[ prob.index( max( prob ) ) ]

    def __extract(
        self : object,
        data : Dict[ Class, Set[ Document ] ],
        pattern : str = '\\b[a-z]{2,}\\b'
        ) -> Dict[ Word, Count ] :
        '''
        Extract Vocabulary from Dataset
        '''
        vocabulary : Dict[ Word, Count ] = {}
        for label in data :
            for document in data[ label ] :
                for word in re.findall(
                    pattern = pattern,
                    string  = document
                    ) :
                    if word not in self.stop_words :
                        if word in vocabulary :
                            vocabulary[ word ] += 1
                        else :
                            vocabulary[ word ] = 1
        return vocabulary

    def __removeFirstQ(
        self : object,
        vocabulary : Dict[ Word, Count ]
        ) -> Dict[ Word, Count ] :
        '''
        Remove First Quartile in Vocabulary
        '''
        if not vocabulary : return vocabulary
        count = sorted( set( vocabulary.values() ) )
        index = len( count ) / 4
        if index % 1 == 0 :
            limit = count[ int( index ) ]
        else :
            index = int( index ) # truncate
            limit = ( count[ index ] + count[ index + 1 ] ) / 2
        for word, count in vocabulary.copy().items() :
            if count < limit :
                del vocabulary[ word ]
        return vocabulary


    def __classVocab(
        self : object,
        data : Dict[ Class, Set[ Document ] ],
        vocabulary : Dict[ Word, Count ]
        ) -> Dict[ Word, Count ] :
        '''
        Extract Vocabulary by Class from Dataset
        '''
        classvocab : Dict[ Word, Count ] = {}
        for label in data :
            classvocab[ label ] = {}
            remaining = vocabulary.copy()
            for document in data[ label ] :
                for word in re.findall( 
                    pattern = '\\b[a-z]{2,}\\b', 
                    string  = document 
                ) :
                    if word in vocabulary :
                        if word in classvocab[ label ] :
                            classvocab[ label ][ word ] += 1
                        else :
                            classvocab[ label ][ word ] = 1
                        if word in remaining :
                            del remaining[ word ]
            for word in remaining :
                classvocab[ label ][ word ] = 0
        return classvocab
