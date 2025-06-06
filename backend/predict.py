import tensorflow as tf
from tokenizer import TashkeelTokenizer, transliterate_text,TashkeelModel,remove_non_arabic
from Cbhgmodel import CBHGModel  
import torch
from pytorch_lightning import LightningModule# ou autre architecture si différente

# Chargement du modèle et du tokenizer
model = tf.keras.models.load_model("C:/Users/pc/application/backend/Model/best_model_CBHG.keras", custom_objects={"CBHGModel": CBHGModel})
tokenizer = TashkeelTokenizer()

def predict_diacritics(text):
    # Encodage du texte
    input_tensor, _ = tokenizer.encode(text, test_match=False)
    input_tensor = tf.expand_dims(input_tensor, axis=0)

    # Prédiction
    prediction = model.predict(input_tensor)
    predicted_ids = tf.argmax(prediction, axis=-1).numpy()[0]

    # Reconstruction des paires (lettre, diacritique)
    decoded = []
    special_tokens = {'<BOS>', '<EOS>', '<PAD>'}

    for letter_idx, tashkeel_idx in zip(input_tensor.numpy()[0], predicted_ids):
        if letter_idx >= len(tokenizer.letters):
            continue
        letter = tokenizer.letters[letter_idx]
        if letter in special_tokens:
            continue
        tashkeel = tokenizer.tashkeel_list[tashkeel_idx] if tashkeel_idx < len(tokenizer.tashkeel_list) else ''
        decoded.append((letter, tashkeel))

    # Reconstruction en Buckwalter
    reconstructed_bw = tokenizer.combine_tashkeel_with_text(decoded)

    # Translittération en arabe
    reconstructed_ar = transliterate_text(reconstructed_bw, direction='bw2ar')

    return reconstructed_ar




###tRansformer


def load_tashkeel_model(ckpt_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = TashkeelTokenizer()

    # Charger l'état du checkpoint pour extraire les hyperparams
    checkpoint = torch.load(ckpt_path, map_location=device)
    hparams = checkpoint.get('hyper_parameters', checkpoint.get('hparams', {}))
    n_layers = hparams.get('n_layers', 6)
    max_seq_len = hparams.get('max_seq_len', 512)

    model = TashkeelModel.load_from_checkpoint(
        ckpt_path,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        n_layers=n_layers,
        learnable_pos_emb=False
    )

    model.eval()  # met le modèle en mode évaluation
    return model

# Fonction de prédiction avec passage du modèle
def tashkeel_text(input_text, model):
    cleaned_text = remove_non_arabic(input_text)

    with torch.no_grad():
        result = model.do_tashkeel_batch([cleaned_text], batch_size=1, verbose=False)

    return result[0]