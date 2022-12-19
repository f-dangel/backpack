cd splash
bundle exec jekyll build -d "../../docs"
cd ..
touch ../docs/.nojekyll
cp CNAME ../docs/CNAME