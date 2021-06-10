echo "=== Acquiring datasets ==="
echo "---"

mkdir -p test_data
mkdir -p test_data/blur

cd test_data/blur

echo "- Downloading WikiText-103 (WT2)"
if [[ ! -d 'wikitext-103' ]]; then
    wget --continue https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
    unzip -q wikitext-103-v1.zip
    cd wikitext-103
    mv wiki.train.tokens train.txt
    mv wiki.valid.tokens valid.txt
    mv wiki.test.tokens test.txt
    cd ..
fi