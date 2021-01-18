import nltk
from summarizer import Summarizer
from transformers import *

get_value = get_value.dict() # данные из request.POST
        
text = get_value['t']
name = get_value['n']
number = get_value['s']
n = int(len(nltk.sent_tokenize(text)))
size1 = int(round(n*(float(number)/100)) + 1)
size = int((n - size1) + 1)
if name == 'gpt':
    name_model = 'sberbank-ai/rugpt3small_based_on_gpt2'
if name == 'bert':
    name_model = 'bert-base-multilingual-cased'
if name == 'rubert':
    name_model = 'DeepPavlov/rubert-base-cased'

custom_config = AutoConfig.from_pretrained(name_model)
custom_config.output_hidden_states = True
custom_tokenizer = AutoTokenizer.from_pretrained(name_model)
custom_model = AutoModel.from_pretrained(name_model, config=custom_config)
model = Summarizer(custom_model=custom_model, custom_tokenizer=custom_tokenizer)

result = model(text, num_sentences=size)
full = ''.join(result)
data = {}
data['result'] = full
