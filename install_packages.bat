conda install -y^
  python=3.12 pytorch torchvision torchaudio pytorch-cuda=12.4 mpi4py jupyter blobfile tqdm matplotlib scikit-learn^
  -c pytorch -c nvidia -c conda-forge

REM `solidspy` is not supported on Windows.
REM conda install -y kssgarcia::solidspy