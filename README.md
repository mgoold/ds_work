
# ds_template

## A repurposing of the Zendesk template for data science docs

### Details:

This repo is a repurposing of Zendesk's Github pages [repo](https://github.com/zendesk/jekyll-theme-zendesk-garden).  It has less total functionality than the original repo, but has added notes to make it a bit more accessible to those who, like me, want to minimize time spent understanding jekyll implementations.

I've already set the remote_theme in the root `_config.yml`, so you should be able to have it work by just setting it up as a github pages site without further coding.

Notes on usage: please see the index file in the docs directory [here](https://github.com/mgoold/ds_template/blob/main/docs/index.md).  It contains the startup content that would normally go in this readme page.

If you're interesting in using this template, it's probably bc of the sidebar and pagination, the details of which can be found [here](https://github.com/mgoold/ds_template/blob/main/docs/customization/sidebar.md).  A test directory to show how nesting works with the sidebar is given at docs/test_nesting.  The sidebar only takes one level of nesting.  I tried to use [recursive liquid code](https://jekyllrb.com/tutorials/navigation/#scenario-9-nested-tree-navigation-with-recursion) to get an arbitrary depth but no joy.  Feel free to have a go yourself. 

I also stripped out the original side matter containing the Zendesk logo.

Happy logging!