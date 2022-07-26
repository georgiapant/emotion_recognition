# project_root_path = r"C:Users\georgiapant\PycharmProjects\GitHub\rebecca"
# project_root_path = "./"
project_root_path = r"C:\Users\georgiapant\PycharmProjects\GitHub\rebecca"
data_path = "D:REBECCA\Datasets\Emotion"

api_url = "http://160.40.51.26:3000" # Browser plugin url
mongodb_parameters = ('localhost', 27017)

MAX_LEN = 126
BATCH_SIZE = 16
BERT_MODEL = 'bert-base-uncased'
RANDOM_SEED = 42
EPOCHS = 100
patience = 3
# bidirectional = True
# coverages = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
alpha = 0.5
mc_dropout_rate = 0
weight_decay = 0.0005
basic_dropout_rate = 0.3
cov = 0.7
lamda = 20 #32

# EKMAN dataset
emotions = ['sadness', 'fear', 'surprise', 'joy', 'disgust', 'anger']

#GoEmotions dataset
# emotions = []

num_labels = len(emotions)