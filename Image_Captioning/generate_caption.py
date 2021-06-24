import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model, load_model


# extract features from each photo in the directory
def extract_features(filename):
    model = VGG19()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature


# map an integer to a word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text


def get_desc(filename):
    model = load_model('./Image_Captioning/model_data/model_weights.h5')
    tokenizer = load(open('./Image_Captioning/model_data/tokenizer.pkl', 'rb'))
    max_length = 34
    photo = extract_features(filename)
    description = generate_desc(model, tokenizer, photo, max_length)
    description = description.split()[1:-1]
    return ' '.join(description)


if __name__ == '__main__':
    description = get_desc('example.jpg')
    print(description)