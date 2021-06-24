# Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pickle import dump
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from tqdm import tqdm


# extract features from each photo in the directory
def extract_features(directory):
    # load the VGG 19 model (weights are pretrained)
    model = VGG19()
    # re-structure the model
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # print(model.summary())
    features = dict()
    for name in tqdm(os.listdir(directory)):
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        image_id = name.split('.')[0]
        features[image_id] = feature
        # print('%s' % name)
    return features


def save_features():
    # extract features from all images
    directory = 'dataset/Flicker8k_Dataset'
    print("Extracting features... (It may take some time.)")
    features = extract_features(directory)
    print('Extracted Features: %d' % len(features))
    dump(features, open('model_data/features.pkl', 'wb'))


if __name__ == '__main__':
    save_features()