#!/bin/bash

mkdir dataset

echo "Downloading Flicker8k_Dataset... (It may take some time, depending on the network speed.)"
wget -q https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip -P dataset/
unzip -q dataset/Flickr8k_Dataset.zip -d dataset
rm -r dataset/Flickr8k_Dataset.zip

wget -q https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip -P dataset/
unzip -q dataset/Flickr8k_text.zip -d dataset
rm -r dataset/Flickr8k_text.zip

rm -r dataset/__MACOSX
echo "DONE."