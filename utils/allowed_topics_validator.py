from sentence_transformers import SentenceTransformer, util
from const.allowed_words import music_study_guitar_study_words

model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model = SentenceTransformer(model_name);

ALLOWED_TOPICS = music_study_guitar_study_words

topic_embeddings = model.encode(ALLOWED_TOPICS, convert_to_tensor=True)

def is_allowed_topic(prompt):
    prompt_embedding = model.encode(prompt, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(prompt_embedding, topic_embeddings)
    
    if max(similarities[0]) > 0.3:
        return True
    else:
        return False