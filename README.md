```bash
git clone https://github.com/farrellhung/flux-finetune.git
cd flux-finetune
git submodule update --init --recursive
python -m venv venv
source venv/bin/activate
pip install torch
pip install -r requirements.txt
huggingface-cli login
```
```bash
python run.py config/test_lora1.yml
```
```bash
python run.py config/07-11-lora-cat-dev.yml && python run.py config/07-11-lora-yoshua-dev.yml && python run.py config/07-11-sborafa-cat-dev.yml && python run.py config/07-11-sborafa-yoshua-dev.yml && python run.py config/07-11-sborafb-cat-dev.yml && python run.py config/07-11-sborafb-yoshua-dev.yml
```


python run.py config/17-11-lora-yoshua-dev-guidance2.yml && python run.py config/17-11-lora-yoshua-dev.yml && python run.py config/17-11-sborafa-yoshua-dev-guidance2.yml && python run.py config/17-11-sborafa-yoshua-dev-lr0.4.yml && python run.py config/17-11-sborafa-yoshua-dev-lr1.yml && python run.py config/17-11-sborafa-yoshua-dev-lr2.yml && python run.py config/17-11-sborafa-yoshua-dev-lr4.yml && python run.py config/17-11-sborafa-yoshua-dev-lr8.yml && python run.py config/17-11-sborafb-yoshua-dev-guidance2.yml && python run.py config/17-11-sborafb-yoshua-dev-lr0.4.yml && python run.py config/17-11-sborafb-yoshua-dev-lr1.yml && python run.py config/17-11-sborafb-yoshua-dev-lr2.yml && python run.py config/17-11-sborafb-yoshua-dev-lr4.yml && python run.py config/17-11-sborafb-yoshua-dev-lr8.yml