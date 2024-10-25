```bash
git clone https://github.com/farrellhung/flux-finetune.git
cd flux-finetune
git submodule update --init --recursive
python -m venv venv
source venv/bin/activate
pip install torch
pip install -r requirements.txt
