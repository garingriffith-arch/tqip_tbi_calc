# Implementation sources

This directory contains the exact analysis implementations executed by the numbered entrypoints in the parent workflow directories.

The project evolved through multiple development iterations before the manuscript specification was fixed. Some preserved implementation files therefore retain historical internal object names, output-directory names, or comments from that development process. Those names are kept here for provenance and to avoid changing executable analysis code solely for presentation.

For manuscript review and reproduction, use the numbered entrypoints documented in `../README.md`. The entrypoints provide a stable 01–17 execution order and describe the role of each analysis in manuscript terminology.

The three larger implementation sources for deployment fitting, manuscript metric generation, and figure generation are stored as gzip-compressed R source files. Their corresponding entrypoints open and source them directly with base R.
