cd splash
bundle exec jekyll build -d "../../docs"
cd ..
sphinx-build "rtd" "../docs/rtd" 
touch ../docs/.nojekyll
touch ../docs/rtd/.nojekyll