**Building the web version**

Requirements: [Jekyll](https://jekyllrb.com/docs/installation/) and [Sphinx](https://www.sphinx-doc.org/en/1.8/usage/installation.html)

Full build to output results in `../docs`
```
bash buildweb.sh
```

Local build of the Jekyll splash page 
```
cd splash
bundle exec jekyll server
```
and go to `localhost:4000/backpack`

Local build of the documentation
```
cd rtd
make
```
and open `/docs_src/rtd_output/index.html`



