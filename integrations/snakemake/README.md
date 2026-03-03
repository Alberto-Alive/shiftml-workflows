# Snakemake integration

This folder provides a minimal wrapper that shells out to `shiftmlwf predict`.

Example rule:

```python
rule predict_shifts:
    input:
        "data/structures.extxyz"
    output:
        directory("results/shiftml")
    shell:
        "shiftmlwf predict {input} --out {output} --workers 4 --cache-dir .cache/shiftml"
```
