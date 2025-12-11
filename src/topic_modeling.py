from bertopic import BERTopic


def fit_bertopic(docs):
    topic_model = BERTopic(language="english")
    topics, probs = topic_model.fit_transform(docs)
    return topic_model, topics, probs
