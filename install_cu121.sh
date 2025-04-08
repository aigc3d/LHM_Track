# install torch 2.3.0
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121

# install base packages
pip install -r requirements.txt

# install pytorch3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# note that this repo uses the sam2 modified by samurai, not the original sam2.
pip install -e ./engine/samurai/sam2
