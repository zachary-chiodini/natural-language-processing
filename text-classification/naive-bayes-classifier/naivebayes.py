import random, re
from math import log
from typing import Dict, List, Set, Union

class NaiveBayes :
    '''
    Multinomial Naive Bayes
    '''
    Count, Probaility = int, float
    Class, Output, Document, Word = str, str, str, str

    def __init__( self ) -> None :
        self.logPc  : Dict[ Class, Probability ] = {} 
        self.logPwc : Dict[ Class, Dict[ Word, Probability ] ] = {}
        self.vocabulary  : Dict[ Class, Set[ Document ] ] = {}
        self.classvocab  : Dict[ Class, Set[ Document ] ] = {}
        self.predictions : Dict[ Class, Dict[ Class, int ] ] = {}
        self.accuracy = 0.0
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
        accuracy = 0.0
        for label in data :
            self.predictions[ label ] = { label : 0 for label in data }
            for document in data[ label ] :
                tot_docs += 1
                output = self.output( document )
                self.predictions[ label ][ output ] += 1
                accuracy += ( label == output )
        if tot_docs :
            self.accuracy = accuracy / tot_docs
        else :
            self.accuracy = 0
        return

    def trainAndTest(
        self  : object,
        data  : Dict[ Class, Set[ Document ] ],
        ratio : float = 0.0,
        iters : int = 10
        ) -> None :
        '''
        Monte Carlo Cross-Validation
        '''
        assert 0 <= ratio < 1, 'Ratio is between 0 and 1.'
        test, train = {}, {}
        ntested, ntrained = 0, 0
        accuracy = []
        truelabelnumb = { label : 0 for label in data }
        predictions = {
            label : { label : 0  for label in data }
            for label in data
            }
        for _ in range( iters ) :
            split = ratio if ratio else random.random()
            for label in data :
                index = int( split * len( data[ label ] ) )
                while not index :
                    split = random.random()
                    index = int( split * len( data[ label ] ) )
                shuffle = list( data[ label ] )
                random.shuffle( shuffle )
                train[ label ] = shuffle[ : index ]
                test [ label ] = shuffle[ index : ]
                truelabelnumb[ label ] += len( test[ label ] )
                ntrained += len( train[ label ] )
                ntested += len( test[ label ] )
            self.train( train )
            self.test( test )
            accuracy.append( self.accuracy )
            for label in self.predictions :
                for predicted in self.predictions[ label ] :
                    predictions[ label ][ predicted ] += \
                        self.predictions[ label ][ predicted ]
        # Calculating and displaying stats
        # True Positive Rate : True Positives / Actual Positives
        # False Positive Rate: False Positive / Actual Negatives
        # Precision: True Positives / ( True Positives + False Positives )
        # Recall   : True Positives / ( True Positives + False Negatives )
        self.predictions = predictions.copy()
        for label in predictions :
            truth_rate = predictions[ label ][ label ] / truelabelnumb[ label ]
            false_rate = (
                sum( predictions[ label_ ][ label ]
                     for label_ in predictions ) \
                - predictions[ label ][ label ]
                ) / sum(
                    predictions[ label ][ output ]
                    for output in predictions
                    )
            precision = predictions[ label ][ label ] / sum(
                predictions[ label_ ][ label ]
                for label_ in predictions
                )
            self.predictions[ label ][ 'precision'  ] = precision
            self.predictions[ label ][ 'truth rate' ] = truth_rate
            self.predictions[ label ][ 'false rate' ] = false_rate
        if len( accuracy ) > 1 :
            mean = sum( accuracy ) / len( accuracy )
            stdv = sum( ( x - mean )**2 / ( len( accuracy ) - 1 )
                         for x in accuracy )**0.5
            deci = self.__decimalPlace( stdv )
        else :
            mean, stdv, deci = self.accuracy, 0, 2
        stdv = round( stdv, deci )
        accuracy = round( mean, deci )
        self.predictions[ 'model' ] = {}
        self.predictions[ 'model' ][ 'accuracy' ] = '{}({})'.format( accuracy, stdv )
        print( 'Examples Trained:', ntrained )
        print( 'Examples Tested :', ntested )
        print( 'Total Examples  :', ntrained + ntested )
        return

    def output( self : object, document : str ) -> Class :
        '''
        Output of Naive Bayes Model with Document Input
        '''
        assert self.logPwc, 'A Naive Bayes model must be trained.'
        result = { label : self.logPc[ label ]
                   for label in self.classvocab }
        for word in re.findall(
            pattern = '\\b[a-z]{2,}\\b',
            string  = document
            ) :
            if word in self.vocabulary :
                for label in self.classvocab :
                    result[ label ] += self.logPwc[ label ][ word ]
        label, prob = zip( *result.items() )
        return label[ prob.index( max( prob ) ) ]

    def __decimalPlace( self, n : Union[ int, str ] ) -> int :
        n = str( n )
        if '.' in n :
            i, f = str( n ).split( '.' )
            if i != '0' :
                return -len( i )
            rslt = 0
            for digit in f :
                rslt += 1
                if digit != '0' :
                    return rslt
        return -len( n )
                    

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
