# adirondax benchmarks

Benchmarking [adirondax](https://github.com/AdirondaxProject/adirondax) with [airspeed velocity](https://asv.readthedocs.io/)

For now, test and upload manually. In this repo folder, do a:

```console
asv run --skip-existing-commits ALL
```

To create a website collating the results, do:

```console
asv publish
```

To view it, do:

```console
asv preview
```

To add the test results to the repo, do a:

```console
git add results/
git commit -m 'adding latest results'
git push
```

To push the published webpage to the `gh-pages` branch on GitHub, do

```console
asv gh-pages --rewrite
```

You can now view the up-to-date [benchmarking page](https://adirondaxproject.github.io/adirondax-benchmarks/).
