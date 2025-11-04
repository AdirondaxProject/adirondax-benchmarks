#!/bin/sh

asv publish
asv preview

open http://127.0.0.1:8080/
