mkdir -p data

wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz -P data/
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz -P data/

cd data
tar -xf images.tar.gz
tar -xf annotations.tar.gz

rm images.tar.gz
rm annotations.tar.gz