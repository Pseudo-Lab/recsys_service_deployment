from langchain_huggingface import HuggingFaceEmbeddings
import torch


def get_embedding_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    embeddings_model = HuggingFaceEmbeddings(
        model_name='upskyy/bge-m3-korean',
        model_kwargs={'device':device},
        encode_kwargs={'normalize_embeddings':False},
    )

    return embeddings_model