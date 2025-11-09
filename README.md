# HVDS Datathon — Iceland Aluminium Exports

End-to-end pipeline to:
- fetch aluminium exports from Hagstofan (PXWeb),
- convert to a tidy quarterly time series,
- train a baseline model (lags only),
- train an upgraded model with drivers (CPI, FX, KEF cargo flights).

---

## 0) Prereqs

- **VS Code** (Python + Jupyter extensions)
- **Conda/Miniconda**

---

## 1) Create & activate the environment

```
conda env create -f environment.yml
conda activate hvds-datathon
python -m ipykernel install --user --name hvds-datathon
```

## 2) Download & parse to parquet run these commands (POWERSHELL COMMANDS, IF YOU ARE USING ANOTHER TERMINAL RUN EACH COMMAND INDIVIDUALLY, DONT USE "`" BEHIND EACH LINE)

# 2a) Download raw JSON from PXWeb
```
python -m src.fetch_pxweb `
  --url   "https://px.hagstofa.is:443/pxen/api/v1/en/Efnahagur/utanrikisverslun/3_voruthjonusta/voruthjonusta/UTA05003.px" `
  --query .\data\queries\aluminium_quarterly.json `
  --out   .\data\raw\aluminium_exports.json
```

# 2b) Parse PXWeb JSON -> tidy quarterly parquet (date,value), 2015Q1+
```
python -m src.parse_pxweb `
  --raw .\data\raw\aluminium_exports.json `
  --out .\data\processed\aluminium_exports.parquet
```

## 3) Train model with drivers (CPI,FX, KEFLAVIK cargo), adjust params as needed
```
python .\src\train.py `       
  --target data/processed/aluminium_exports.parquet `   
  --drivers_dir Hagstofan-Datathon-data-2025-main `
  --out_dir outputs/run ` 
  --early_stop 20 `
  --test-frac 0.2 `
  --seed 42
```


## LLM usage and additional tools:
ChatGpt was used to a large extent for help with data parsing as some of the csv files used were corrupted, as well as data normalization for use with multiple datasets as many were incompatible in their given form.
There were also a lot of version mismatches/UTF-8 errors/incompatible data, so LLM's were used for try -> catch statements to identify problems, couldn't get around to removing them because of time constraints.
Originally the train.py script structure was a jangled mess that was hard to read and functions became way too arduous to read and inefficient, a better script layout was given as suggested by GPT and I followed that structure to some extent.

I even fed the actual data I used in this project to an advanced LLM model for it to make a similar prediction model to me for comparison with my actual project, and with that I could conclude that my data prediction model makes sense.

Numerous stack overflow threads were used for this project.
Well as the provided github scripts made by the datathon teachers/hosts.


https://www.ibm.com/think/topics/predictive-ai
https://www.youtube.com/watch?v=rcXAb1eKZ_I


For any additional queries or questions relating to this project please mail me via oss27@hi.is