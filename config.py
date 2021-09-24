class Config:
    def __init__(self,
                 chunk_length=3,  # in seconds
                 sample_rate=16000,  # Hz
                 n_samples=-1,  # number of audio files will be used to generate training features. -1 means all files
                 raw_data_corpus='./csv/new2.csv',
                 clean_data_corpus='./csv/clean_data_corpus.csv',

                 raw_data_dir='./raw_audio',
                 clean_data_dir='./clean_audio',
                 pickle_dir='./pickle', checkpoint_dir='./checkpoint'):

        # Sleepy threshold
        self.sleepy_threshold = 4 # 4,5,6,7

        # Corpus files
        self.raw_data_corpus = raw_data_corpus
        self.clean_data_corpus = clean_data_corpus

        # Directories
        self.raw_data_dir = raw_data_dir
        self.clean_data_dir = clean_data_dir
        self.pickle_dir = pickle_dir
        self.checkpoint_dir = checkpoint_dir

        self.NUMBER_TRAINING_SAMPLES = n_samples  # number of training sample
        self.NUMBER_TESTING_SAMPLES = 5000  # number of training sample
        self.SAMPLE_RATE = sample_rate  # the sample rate of training audio files

        # Chunks' size
        self.CHUNK_LENGTH = chunk_length  # in seconds (default is 3 seconds)
        self.CHUNK_SIZE = self.CHUNK_LENGTH * self.SAMPLE_RATE
        self.MFCC_CHUNK_SIZE = int(
            0.25 * self.SAMPLE_RATE)  # mfcc & spectrogram features, each chunk's length is 0.25ms
        self.SPECTROGRAM_CHUNK_SIZE = int(0.25 * self.SAMPLE_RATE)

        # The serialization process crashes when (Chunk_Length=3sec and NUMBER_TRAINING_SAMPLES > 5000)
        # In order to avoid crashing, we will save the data in multiple files.
        # This variable stores the index-offsets of data-blocks in pickle files
        #   e.g. [0, 2356, 9130] means the output pickle file will be stored in 3 sub-files.
        #       file1: features[0:2355]
        #       file2: features[2356:9129]
        #       file3: features[9130:]
        # This variable is only used when generating training feature from audio files.
        # When loading from pickle files, the length of the file1 is implied file2's offset
        self.PICKLE_OFFSET_INDEXES = []
        self.PICKlE_FILES_THRESHOLD = 2000       # this means we separate pickle file every 2000 audio files.
                                                 # I tried with 3000, it crashed

# --------------------------------------------------------------------------------------
__cfg__ = Config(raw_data_dir='/media/data_sdf/sessionsdata',
                 raw_data_corpus='/media/data_sdf/bang/SleepinessDetection/new2-v1.csv')
