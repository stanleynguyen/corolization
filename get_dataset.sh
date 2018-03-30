echo "Downloading SUN dataset..."
wget -c http://groups.csail.mit.edu/vision/SUN/releases/SUN2012.tar.gz
echo "Extracting..."
tar xzf SUN2012.tar.gz
echo "Resizing images..."
find . -name "*.jpg" | xargs mogrify -resize 256x256!
echo "Done!"

