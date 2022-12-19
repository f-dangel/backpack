**Building the web version**

Requirements: [Jekyll](https://jekyllrb.com/docs/installation/) and [Sphinx](https://www.sphinx-doc.org/en/1.8/usage/installation.html) 
and installing the jekyll dependencies (`bundle install` in `docs_src/splash`)

- Full build to output results in `../docs`
  ```
  bash buildweb.sh
  ```

- Local build of the Jekyll splash page 
  ```
  cd splash
  bundle exec jekyll server
  ```
  and go to `localhost:4000/backpack`
  
  Note: The code examples on backpack.pt are defined with HTML tags in 
  `splash/_includes/code-samples.html`. 
  The python files in `splash/_includes/` are copy-pasted from the resulting page
  for testing purposes. Editing the tags in the HTML file is manual. 

- Local build of the documentation
  ```
  cd rtd
  make
  ```
  and open `/docs_src/rtd_output/index.html`



