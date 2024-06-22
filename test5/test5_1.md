# 实验5_1

## 1.预备工作

### 首先安装程序运行必备的一些库。

```python
!pip install tflite-model-maker
```

    Collecting tflite-model-maker
      Downloading tflite_model_maker-0.4.3-py3-none-any.whl.metadata (5.4 kB)
    Collecting tf-models-official==2.3.0 (from tflite-model-maker)
      Downloading tf_models_official-2.3.0-py2.py3-none-any.whl.metadata (1.3 kB)
    Collecting numpy<1.23.4,>=1.17.3 (from tflite-model-maker)
      Downloading numpy-1.23.3-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.3 kB)
    Collecting pillow>=7.0.0 (from tflite-model-maker)
      Downloading pillow-10.3.0-cp38-cp38-manylinux_2_28_x86_64.whl.metadata (9.2 kB)
    Collecting sentencepiece>=0.1.91 (from tflite-model-maker)
      Downloading sentencepiece-0.2.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.7 kB)
    Collecting tensorflow-datasets>=2.1.0 (from tflite-model-maker)
      Downloading tensorflow_datasets-4.9.2-py3-none-any.whl.metadata (9.0 kB)
    Collecting fire>=0.3.1 (from tflite-model-maker)
      Downloading fire-0.6.0.tar.gz (88 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m88.4/88.4 kB[0m [31m1.9 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25h  Preparing metadata (setup.py) ... [?25ldone
    [?25hCollecting flatbuffers>=2.0 (from tflite-model-maker)
      Downloading flatbuffers-24.3.25-py2.py3-none-any.whl.metadata (850 bytes)
    Collecting absl-py>=0.10.0 (from tflite-model-maker)
      Downloading absl_py-2.1.0-py3-none-any.whl.metadata (2.3 kB)
    Collecting urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 (from tflite-model-maker)
      Downloading urllib3-1.25.11-py2.py3-none-any.whl.metadata (41 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m41.1/41.1 kB[0m [31m1.0 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting tflite-support>=0.4.2 (from tflite-model-maker)
      Downloading tflite_support-0.4.3-cp38-cp38-manylinux2014_x86_64.whl.metadata (2.4 kB)
    Collecting tensorflowjs<3.19.0,>=2.4.0 (from tflite-model-maker)
      Downloading tensorflowjs-3.18.0-py3-none-any.whl.metadata (1.6 kB)
    Collecting tensorflow>=2.6.0 (from tflite-model-maker)
      Downloading tensorflow-2.13.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.4 kB)
    Collecting numba>=0.53 (from tflite-model-maker)
      Downloading numba-0.58.1-cp38-cp38-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (2.7 kB)
    Collecting librosa==0.8.1 (from tflite-model-maker)
      Downloading librosa-0.8.1-py3-none-any.whl.metadata (6.8 kB)
    Collecting lxml>=4.6.1 (from tflite-model-maker)
      Downloading lxml-5.2.2-cp38-cp38-manylinux_2_28_x86_64.whl.metadata (3.4 kB)
    Collecting PyYAML>=5.1 (from tflite-model-maker)
      Downloading PyYAML-6.0.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.1 kB)
    Collecting matplotlib<3.5.0,>=3.0.3 (from tflite-model-maker)
      Downloading matplotlib-3.4.3-cp38-cp38-manylinux1_x86_64.whl.metadata (5.7 kB)
    Requirement already satisfied: six>=1.12.0 in /workspaces/codespaces-jupyter/.conda/lib/python3.8/site-packages (from tflite-model-maker) (1.16.0)
    Collecting tensorflow-addons>=0.11.2 (from tflite-model-maker)
      Downloading tensorflow_addons-0.21.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (1.8 kB)
    Collecting neural-structured-learning>=1.3.1 (from tflite-model-maker)
      Downloading neural_structured_learning-1.4.0-py2.py3-none-any.whl.metadata (2.5 kB)
    Collecting tensorflow-model-optimization>=0.5 (from tflite-model-maker)
      Downloading tensorflow_model_optimization-0.8.0-py2.py3-none-any.whl.metadata (904 bytes)
    Collecting Cython>=0.29.13 (from tflite-model-maker)
      Downloading Cython-3.0.10-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.2 kB)
    Collecting scann==1.2.6 (from tflite-model-maker)
      Downloading scann-1.2.6-cp38-cp38-manylinux2014_x86_64.whl.metadata (4.5 kB)
    Collecting tensorflow-hub<0.13,>=0.7.0 (from tflite-model-maker)
      Downloading tensorflow_hub-0.12.0-py2.py3-none-any.whl.metadata (1.7 kB)
    Collecting audioread>=2.0.0 (from librosa==0.8.1->tflite-model-maker)
      Downloading audioread-3.0.1-py3-none-any.whl.metadata (8.4 kB)
    Collecting scipy>=1.0.0 (from librosa==0.8.1->tflite-model-maker)
      Downloading scipy-1.10.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (58 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m58.9/58.9 kB[0m [31m1.6 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting scikit-learn!=0.19.0,>=0.14.0 (from librosa==0.8.1->tflite-model-maker)
      Downloading scikit_learn-1.3.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (11 kB)
    Collecting joblib>=0.14 (from librosa==0.8.1->tflite-model-maker)
      Downloading joblib-1.4.2-py3-none-any.whl.metadata (5.4 kB)
    Requirement already satisfied: decorator>=3.0.0 in /workspaces/codespaces-jupyter/.conda/lib/python3.8/site-packages (from librosa==0.8.1->tflite-model-maker) (5.1.1)
    Collecting resampy>=0.2.2 (from librosa==0.8.1->tflite-model-maker)
      Downloading resampy-0.4.3-py3-none-any.whl.metadata (3.0 kB)
    Collecting soundfile>=0.10.2 (from librosa==0.8.1->tflite-model-maker)
      Downloading soundfile-0.12.1-py2.py3-none-manylinux_2_31_x86_64.whl.metadata (14 kB)
    Collecting pooch>=1.0 (from librosa==0.8.1->tflite-model-maker)
      Downloading pooch-1.8.2-py3-none-any.whl.metadata (10 kB)
    Requirement already satisfied: packaging>=20.0 in /workspaces/codespaces-jupyter/.conda/lib/python3.8/site-packages (from librosa==0.8.1->tflite-model-maker) (24.1)
    Collecting tensorflow>=2.6.0 (from tflite-model-maker)
      Downloading tensorflow-2.8.4-cp38-cp38-manylinux2010_x86_64.whl.metadata (2.9 kB)
    Collecting dataclasses (from tf-models-official==2.3.0->tflite-model-maker)
      Downloading dataclasses-0.6-py3-none-any.whl.metadata (3.0 kB)
    Collecting gin-config (from tf-models-official==2.3.0->tflite-model-maker)
      Downloading gin_config-0.5.0-py3-none-any.whl.metadata (2.9 kB)
    Collecting google-api-python-client>=1.6.7 (from tf-models-official==2.3.0->tflite-model-maker)
      Downloading google_api_python_client-2.133.0-py2.py3-none-any.whl.metadata (6.7 kB)
    Collecting google-cloud-bigquery>=0.31.0 (from tf-models-official==2.3.0->tflite-model-maker)
      Downloading google_cloud_bigquery-3.24.0-py2.py3-none-any.whl.metadata (8.9 kB)
    Collecting kaggle>=1.3.9 (from tf-models-official==2.3.0->tflite-model-maker)
      Downloading kaggle-1.6.14.tar.gz (82 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m82.1/82.1 kB[0m [31m2.5 MB/s[0m eta [36m0:00:00[0m
    [?25h  Preparing metadata (setup.py) ... [?25ldone
    [?25hCollecting opencv-python-headless (from tf-models-official==2.3.0->tflite-model-maker)
      Downloading opencv_python_headless-4.10.0.82-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (20 kB)
    Collecting pandas>=0.22.0 (from tf-models-official==2.3.0->tflite-model-maker)
      Downloading pandas-2.0.3-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (18 kB)
    Requirement already satisfied: psutil>=5.4.3 in /workspaces/codespaces-jupyter/.conda/lib/python3.8/site-packages (from tf-models-official==2.3.0->tflite-model-maker) (5.9.8)
    Collecting py-cpuinfo>=3.3.0 (from tf-models-official==2.3.0->tflite-model-maker)
      Downloading py_cpuinfo-9.0.0-py3-none-any.whl.metadata (794 bytes)
    Collecting tf-slim>=1.1.0 (from tf-models-official==2.3.0->tflite-model-maker)
      Downloading tf_slim-1.1.0-py2.py3-none-any.whl.metadata (1.6 kB)
    Collecting termcolor (from fire>=0.3.1->tflite-model-maker)
      Downloading termcolor-2.4.0-py3-none-any.whl.metadata (6.1 kB)
    Collecting cycler>=0.10 (from matplotlib<3.5.0,>=3.0.3->tflite-model-maker)
      Downloading cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)
    Collecting kiwisolver>=1.0.1 (from matplotlib<3.5.0,>=3.0.3->tflite-model-maker)
      Downloading kiwisolver-1.4.5-cp38-cp38-manylinux_2_5_x86_64.manylinux1_x86_64.whl.metadata (6.4 kB)
    Collecting pyparsing>=2.2.1 (from matplotlib<3.5.0,>=3.0.3->tflite-model-maker)
      Downloading pyparsing-3.1.2-py3-none-any.whl.metadata (5.1 kB)
    Requirement already satisfied: python-dateutil>=2.7 in /workspaces/codespaces-jupyter/.conda/lib/python3.8/site-packages (from matplotlib<3.5.0,>=3.0.3->tflite-model-maker) (2.9.0)
    Collecting attrs (from neural-structured-learning>=1.3.1->tflite-model-maker)
      Downloading attrs-23.2.0-py3-none-any.whl.metadata (9.5 kB)
    Collecting llvmlite<0.42,>=0.41.0dev0 (from numba>=0.53->tflite-model-maker)
      Downloading llvmlite-0.41.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.8 kB)
    Requirement already satisfied: importlib-metadata in /workspaces/codespaces-jupyter/.conda/lib/python3.8/site-packages (from numba>=0.53->tflite-model-maker) (7.1.0)
    Collecting astunparse>=1.6.0 (from tensorflow>=2.6.0->tflite-model-maker)
      Downloading astunparse-1.6.3-py2.py3-none-any.whl.metadata (4.4 kB)
    Collecting gast>=0.2.1 (from tensorflow>=2.6.0->tflite-model-maker)
      Downloading gast-0.5.4-py3-none-any.whl.metadata (1.3 kB)
    Collecting google-pasta>=0.1.1 (from tensorflow>=2.6.0->tflite-model-maker)
      Downloading google_pasta-0.2.0-py3-none-any.whl.metadata (814 bytes)
    Collecting h5py>=2.9.0 (from tensorflow>=2.6.0->tflite-model-maker)
      Downloading h5py-3.11.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.5 kB)
    Collecting keras-preprocessing>=1.1.1 (from tensorflow>=2.6.0->tflite-model-maker)
      Downloading Keras_Preprocessing-1.1.2-py2.py3-none-any.whl.metadata (1.9 kB)
    Collecting libclang>=9.0.1 (from tensorflow>=2.6.0->tflite-model-maker)
      Downloading libclang-18.1.1-py2.py3-none-manylinux2010_x86_64.whl.metadata (5.2 kB)
    Collecting opt-einsum>=2.3.2 (from tensorflow>=2.6.0->tflite-model-maker)
      Downloading opt_einsum-3.3.0-py3-none-any.whl.metadata (6.5 kB)
    Collecting protobuf<3.20,>=3.9.2 (from tensorflow>=2.6.0->tflite-model-maker)
      Downloading protobuf-3.19.6-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (787 bytes)
    Requirement already satisfied: setuptools in /workspaces/codespaces-jupyter/.conda/lib/python3.8/site-packages (from tensorflow>=2.6.0->tflite-model-maker) (69.5.1)
    Requirement already satisfied: typing-extensions>=3.6.6 in /workspaces/codespaces-jupyter/.conda/lib/python3.8/site-packages (from tensorflow>=2.6.0->tflite-model-maker) (4.12.2)
    Collecting wrapt>=1.11.0 (from tensorflow>=2.6.0->tflite-model-maker)
      Downloading wrapt-1.16.0-cp38-cp38-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.6 kB)
    Collecting tensorboard<2.9,>=2.8 (from tensorflow>=2.6.0->tflite-model-maker)
      Downloading tensorboard-2.8.0-py3-none-any.whl.metadata (1.9 kB)
    Collecting tensorflow-estimator<2.9,>=2.8 (from tensorflow>=2.6.0->tflite-model-maker)
      Downloading tensorflow_estimator-2.8.0-py2.py3-none-any.whl.metadata (1.3 kB)
    Collecting keras<2.9,>=2.8.0rc0 (from tensorflow>=2.6.0->tflite-model-maker)
      Downloading keras-2.8.0-py2.py3-none-any.whl.metadata (1.3 kB)
    Collecting tensorflow-io-gcs-filesystem>=0.23.1 (from tensorflow>=2.6.0->tflite-model-maker)
      Downloading tensorflow_io_gcs_filesystem-0.34.0-cp38-cp38-manylinux_2_12_x86_64.manylinux2010_x86_64.whl.metadata (14 kB)
    Collecting grpcio<2.0,>=1.24.3 (from tensorflow>=2.6.0->tflite-model-maker)
      Downloading grpcio-1.64.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.3 kB)
    Collecting typeguard<3.0.0,>=2.7 (from tensorflow-addons>=0.11.2->tflite-model-maker)
      Downloading typeguard-2.13.3-py3-none-any.whl.metadata (3.6 kB)
    Collecting array-record (from tensorflow-datasets>=2.1.0->tflite-model-maker)
      Downloading array_record-0.4.0-py38-none-any.whl.metadata (502 bytes)
    Collecting click (from tensorflow-datasets>=2.1.0->tflite-model-maker)
      Downloading click-8.1.7-py3-none-any.whl.metadata (3.0 kB)
    Collecting dm-tree (from tensorflow-datasets>=2.1.0->tflite-model-maker)
      Downloading dm_tree-0.1.8-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (1.9 kB)
    Collecting etils>=0.9.0 (from etils[enp,epath]>=0.9.0->tensorflow-datasets>=2.1.0->tflite-model-maker)
      Downloading etils-1.3.0-py3-none-any.whl.metadata (5.5 kB)
    Collecting promise (from tensorflow-datasets>=2.1.0->tflite-model-maker)
      Downloading promise-2.3.tar.gz (19 kB)
      Preparing metadata (setup.py) ... [?25ldone
    [?25hINFO: pip is looking at multiple versions of tensorflow-datasets to determine which version is compatible with other requirements. This could take a while.
    Collecting tensorflow-datasets>=2.1.0 (from tflite-model-maker)
      Downloading tensorflow_datasets-4.9.1-py3-none-any.whl.metadata (9.0 kB)
      Downloading tensorflow_datasets-4.9.0-py3-none-any.whl.metadata (9.0 kB)
    Collecting requests>=2.19.0 (from tensorflow-datasets>=2.1.0->tflite-model-maker)
      Downloading requests-2.32.3-py3-none-any.whl.metadata (4.6 kB)
    Collecting tensorflow-metadata (from tensorflow-datasets>=2.1.0->tflite-model-maker)
      Downloading tensorflow_metadata-1.14.0-py3-none-any.whl.metadata (2.1 kB)
    Collecting toml (from tensorflow-datasets>=2.1.0->tflite-model-maker)
      Downloading toml-0.10.2-py2.py3-none-any.whl.metadata (7.1 kB)
    Collecting tqdm (from tensorflow-datasets>=2.1.0->tflite-model-maker)
      Using cached tqdm-4.66.4-py3-none-any.whl.metadata (57 kB)
    Collecting importlib-resources (from tensorflow-datasets>=2.1.0->tflite-model-maker)
      Downloading importlib_resources-6.4.0-py3-none-any.whl.metadata (3.9 kB)
    Collecting absl-py>=0.10.0 (from tflite-model-maker)
      Downloading absl_py-1.4.0-py3-none-any.whl.metadata (2.3 kB)
    Collecting packaging>=20.0 (from librosa==0.8.1->tflite-model-maker)
      Downloading packaging-20.9-py2.py3-none-any.whl.metadata (13 kB)
    Collecting sounddevice>=0.4.4 (from tflite-support>=0.4.2->tflite-model-maker)
      Downloading sounddevice-0.4.7-py3-none-any.whl.metadata (1.4 kB)
    Collecting pybind11>=2.6.0 (from tflite-support>=0.4.2->tflite-model-maker)
      Downloading pybind11-2.12.0-py3-none-any.whl.metadata (9.5 kB)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in /workspaces/codespaces-jupyter/.conda/lib/python3.8/site-packages (from astunparse>=1.6.0->tensorflow>=2.6.0->tflite-model-maker) (0.43.0)
    Requirement already satisfied: zipp in /workspaces/codespaces-jupyter/.conda/lib/python3.8/site-packages (from etils[enp,epath]>=0.9.0->tensorflow-datasets>=2.1.0->tflite-model-maker) (3.19.2)
    Collecting httplib2<1.dev0,>=0.19.0 (from google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker)
      Downloading httplib2-0.22.0-py3-none-any.whl.metadata (2.6 kB)
    Collecting google-auth!=2.24.0,!=2.25.0,<3.0.0.dev0,>=1.32.0 (from google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker)
      Downloading google_auth-2.30.0-py2.py3-none-any.whl.metadata (4.7 kB)
    Collecting google-auth-httplib2<1.0.0,>=0.2.0 (from google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker)
      Downloading google_auth_httplib2-0.2.0-py2.py3-none-any.whl.metadata (2.2 kB)
    Collecting google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.0,<3.0.0.dev0,>=1.31.5 (from google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker)
      Downloading google_api_core-2.19.0-py3-none-any.whl.metadata (2.7 kB)
    Collecting uritemplate<5,>=3.0.1 (from google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker)
      Downloading uritemplate-4.1.1-py2.py3-none-any.whl.metadata (2.9 kB)
    Collecting google-cloud-core<3.0.0dev,>=1.6.0 (from google-cloud-bigquery>=0.31.0->tf-models-official==2.3.0->tflite-model-maker)
      Downloading google_cloud_core-2.4.1-py2.py3-none-any.whl.metadata (2.7 kB)
    Collecting google-resumable-media<3.0dev,>=0.6.0 (from google-cloud-bigquery>=0.31.0->tf-models-official==2.3.0->tflite-model-maker)
      Downloading google_resumable_media-2.7.1-py2.py3-none-any.whl.metadata (2.2 kB)
    Collecting certifi>=2023.7.22 (from kaggle>=1.3.9->tf-models-official==2.3.0->tflite-model-maker)
      Downloading certifi-2024.6.2-py3-none-any.whl.metadata (2.2 kB)
    Collecting python-slugify (from kaggle>=1.3.9->tf-models-official==2.3.0->tflite-model-maker)
      Downloading python_slugify-8.0.4-py2.py3-none-any.whl.metadata (8.5 kB)
    Collecting bleach (from kaggle>=1.3.9->tf-models-official==2.3.0->tflite-model-maker)
      Downloading bleach-6.1.0-py3-none-any.whl.metadata (30 kB)
    Collecting pytz>=2020.1 (from pandas>=0.22.0->tf-models-official==2.3.0->tflite-model-maker)
      Downloading pytz-2024.1-py2.py3-none-any.whl.metadata (22 kB)
    Collecting tzdata>=2022.1 (from pandas>=0.22.0->tf-models-official==2.3.0->tflite-model-maker)
      Downloading tzdata-2024.1-py2.py3-none-any.whl.metadata (1.4 kB)
    Requirement already satisfied: platformdirs>=2.5.0 in /workspaces/codespaces-jupyter/.conda/lib/python3.8/site-packages (from pooch>=1.0->librosa==0.8.1->tflite-model-maker) (4.2.2)
    Collecting charset-normalizer<4,>=2 (from requests>=2.19.0->tensorflow-datasets>=2.1.0->tflite-model-maker)
      Downloading charset_normalizer-3.3.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (33 kB)
    Collecting idna<4,>=2.5 (from requests>=2.19.0->tensorflow-datasets>=2.1.0->tflite-model-maker)
      Downloading idna-3.7-py3-none-any.whl.metadata (9.9 kB)
    Collecting threadpoolctl>=2.0.0 (from scikit-learn!=0.19.0,>=0.14.0->librosa==0.8.1->tflite-model-maker)
      Downloading threadpoolctl-3.5.0-py3-none-any.whl.metadata (13 kB)
    Collecting CFFI>=1.0 (from sounddevice>=0.4.4->tflite-support>=0.4.2->tflite-model-maker)
      Downloading cffi-1.16.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (1.5 kB)
    Collecting google-auth-oauthlib<0.5,>=0.4.1 (from tensorboard<2.9,>=2.8->tensorflow>=2.6.0->tflite-model-maker)
      Downloading google_auth_oauthlib-0.4.6-py2.py3-none-any.whl.metadata (2.7 kB)
    Collecting markdown>=2.6.8 (from tensorboard<2.9,>=2.8->tensorflow>=2.6.0->tflite-model-maker)
      Downloading Markdown-3.6-py3-none-any.whl.metadata (7.0 kB)
    Collecting tensorboard-data-server<0.7.0,>=0.6.0 (from tensorboard<2.9,>=2.8->tensorflow>=2.6.0->tflite-model-maker)
      Downloading tensorboard_data_server-0.6.1-py3-none-manylinux2010_x86_64.whl.metadata (1.1 kB)
    Collecting tensorboard-plugin-wit>=1.6.0 (from tensorboard<2.9,>=2.8->tensorflow>=2.6.0->tflite-model-maker)
      Downloading tensorboard_plugin_wit-1.8.1-py3-none-any.whl.metadata (873 bytes)
    Collecting werkzeug>=0.11.15 (from tensorboard<2.9,>=2.8->tensorflow>=2.6.0->tflite-model-maker)
      Downloading werkzeug-3.0.3-py3-none-any.whl.metadata (3.7 kB)
    Collecting googleapis-common-protos<2,>=1.52.0 (from tensorflow-metadata->tensorflow-datasets>=2.1.0->tflite-model-maker)
      Downloading googleapis_common_protos-1.63.1-py2.py3-none-any.whl.metadata (1.5 kB)
    INFO: pip is looking at multiple versions of tensorflow-metadata to determine which version is compatible with other requirements. This could take a while.
    Collecting tensorflow-metadata (from tensorflow-datasets>=2.1.0->tflite-model-maker)
      Downloading tensorflow_metadata-1.13.1-py3-none-any.whl.metadata (2.1 kB)
      Downloading tensorflow_metadata-1.13.0-py3-none-any.whl.metadata (2.1 kB)
    Collecting pycparser (from CFFI>=1.0->sounddevice>=0.4.4->tflite-support>=0.4.2->tflite-model-maker)
      Downloading pycparser-2.22-py3-none-any.whl.metadata (943 bytes)
    Collecting proto-plus<2.0.0dev,>=1.22.3 (from google-api-core!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.0,<3.0.0.dev0,>=1.31.5->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker)
      Downloading proto_plus-1.23.0-py3-none-any.whl.metadata (2.2 kB)
    Collecting grpcio-status<2.0.dev0,>=1.33.2 (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-cloud-bigquery>=0.31.0->tf-models-official==2.3.0->tflite-model-maker)
      Downloading grpcio_status-1.64.1-py3-none-any.whl.metadata (1.1 kB)
    Collecting cachetools<6.0,>=2.0.0 (from google-auth!=2.24.0,!=2.25.0,<3.0.0.dev0,>=1.32.0->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker)
      Downloading cachetools-5.3.3-py3-none-any.whl.metadata (5.3 kB)
    Collecting pyasn1-modules>=0.2.1 (from google-auth!=2.24.0,!=2.25.0,<3.0.0.dev0,>=1.32.0->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker)
      Downloading pyasn1_modules-0.4.0-py3-none-any.whl.metadata (3.4 kB)
    Collecting rsa<5,>=3.1.4 (from google-auth!=2.24.0,!=2.25.0,<3.0.0.dev0,>=1.32.0->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker)
      Downloading rsa-4.9-py3-none-any.whl.metadata (4.2 kB)
    Collecting requests-oauthlib>=0.7.0 (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.9,>=2.8->tensorflow>=2.6.0->tflite-model-maker)
      Downloading requests_oauthlib-2.0.0-py2.py3-none-any.whl.metadata (11 kB)
    Collecting google-crc32c<2.0dev,>=1.0 (from google-resumable-media<3.0dev,>=0.6.0->google-cloud-bigquery>=0.31.0->tf-models-official==2.3.0->tflite-model-maker)
      Downloading google_crc32c-1.5.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.3 kB)
    Collecting MarkupSafe>=2.1.1 (from werkzeug>=0.11.15->tensorboard<2.9,>=2.8->tensorflow>=2.6.0->tflite-model-maker)
      Downloading MarkupSafe-2.1.5-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.0 kB)
    Collecting webencodings (from bleach->kaggle>=1.3.9->tf-models-official==2.3.0->tflite-model-maker)
      Downloading webencodings-0.5.1-py2.py3-none-any.whl.metadata (2.1 kB)
    Collecting text-unidecode>=1.3 (from python-slugify->kaggle>=1.3.9->tf-models-official==2.3.0->tflite-model-maker)
      Downloading text_unidecode-1.3-py2.py3-none-any.whl.metadata (2.4 kB)
    INFO: pip is looking at multiple versions of grpcio-status to determine which version is compatible with other requirements. This could take a while.
    Collecting grpcio-status<2.0.dev0,>=1.33.2 (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-cloud-bigquery>=0.31.0->tf-models-official==2.3.0->tflite-model-maker)
      Downloading grpcio_status-1.64.0-py3-none-any.whl.metadata (1.1 kB)
      Downloading grpcio_status-1.63.0-py3-none-any.whl.metadata (1.1 kB)
      Downloading grpcio_status-1.62.2-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.62.1-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.62.0-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.60.1-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.60.0-py3-none-any.whl.metadata (1.3 kB)
    INFO: pip is still looking at multiple versions of grpcio-status to determine which version is compatible with other requirements. This could take a while.
      Downloading grpcio_status-1.59.3-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.59.2-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.59.0-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.58.0-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.57.0-py3-none-any.whl.metadata (1.2 kB)
    INFO: This is taking longer than usual. You might need to provide the dependency resolver with stricter constraints to reduce runtime. See https://pip.pypa.io/warnings/backtracking for guidance. If you want to abort this run, press Ctrl + C.
      Downloading grpcio_status-1.56.2-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.56.0-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.55.3-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.54.3-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.54.2-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.54.0-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.53.2-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.53.1-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.53.0-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.51.3-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.51.1-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.50.0-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.49.1-py3-none-any.whl.metadata (1.3 kB)
      Downloading grpcio_status-1.48.2-py3-none-any.whl.metadata (1.2 kB)
    Collecting pyasn1<0.7.0,>=0.4.6 (from pyasn1-modules>=0.2.1->google-auth!=2.24.0,!=2.25.0,<3.0.0.dev0,>=1.32.0->google-api-python-client>=1.6.7->tf-models-official==2.3.0->tflite-model-maker)
      Downloading pyasn1-0.6.0-py2.py3-none-any.whl.metadata (8.3 kB)
    Collecting oauthlib>=3.0.0 (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.9,>=2.8->tensorflow>=2.6.0->tflite-model-maker)
      Downloading oauthlib-3.2.2-py3-none-any.whl.metadata (7.5 kB)
    Downloading tflite_model_maker-0.4.3-py3-none-any.whl (580 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m580.1/580.1 kB[0m [31m10.2 MB/s[0m eta [36m0:00:00[0m00:01[0m
    [?25hDownloading librosa-0.8.1-py3-none-any.whl (203 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m203.8/203.8 kB[0m [31m6.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading scann-1.2.6-cp38-cp38-manylinux2014_x86_64.whl (10.9 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m10.9/10.9 MB[0m [31m64.7 MB/s[0m eta [36m0:00:00[0m:00:01[0m00:01[0m
    [?25hDownloading tf_models_official-2.3.0-py2.py3-none-any.whl (840 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m840.9/840.9 kB[0m [31m20.1 MB/s[0m eta [36m0:00:00[0m00:01[0m
    [?25hDownloading Cython-3.0.10-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.6 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.6/3.6 MB[0m [31m49.4 MB/s[0m eta [36m0:00:00[0m:00:01[0m
    [?25hDownloading flatbuffers-24.3.25-py2.py3-none-any.whl (26 kB)
    Downloading lxml-5.2.2-cp38-cp38-manylinux_2_28_x86_64.whl (5.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.1/5.1 MB[0m [31m56.2 MB/s[0m eta [36m0:00:00[0m:00:01[0m
    [?25hDownloading matplotlib-3.4.3-cp38-cp38-manylinux1_x86_64.whl (10.3 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m10.3/10.3 MB[0m [31m69.8 MB/s[0m eta [36m0:00:00[0m:00:01[0m0:01[0m
    [?25hDownloading neural_structured_learning-1.4.0-py2.py3-none-any.whl (128 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m128.6/128.6 kB[0m [31m3.3 MB/s[0m eta [36m0:00:00[0mta [36m0:00:01[0m
    [?25hDownloading numba-0.58.1-cp38-cp38-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (3.7 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.7/3.7 MB[0m [31m51.1 MB/s[0m eta [36m0:00:00[0m:00:01[0m
    [?25hDownloading numpy-1.23.3-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (17.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m17.1/17.1 MB[0m [31m56.4 MB/s[0m eta [36m0:00:00[0m:00:01[0m00:01[0m
    [?25hDownloading pillow-10.3.0-cp38-cp38-manylinux_2_28_x86_64.whl (4.5 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m4.5/4.5 MB[0m [31m54.7 MB/s[0m eta [36m0:00:00[0m:00:01[0m
    [?25hDownloading PyYAML-6.0.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (736 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m736.6/736.6 kB[0m [31m18.5 MB/s[0m eta [36m0:00:00[0m00:01[0m
    [?25hDownloading sentencepiece-0.2.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.3 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.3/1.3 MB[0m [31m25.4 MB/s[0m eta [36m0:00:00[0m:00:01[0m
    [?25hDownloading tensorflow-2.8.4-cp38-cp38-manylinux2010_x86_64.whl (498.0 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m498.0/498.0 MB[0m [31m4.0 MB/s[0m eta [36m0:00:00[0m:00:01[0m00:01[0m
    [?25hDownloading tensorflow_addons-0.21.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (612 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m612.0/612.0 kB[0m [31m14.9 MB/s[0m eta [36m0:00:00[0m00:01[0m
    [?25hDownloading tensorflow_datasets-4.9.0-py3-none-any.whl (5.4 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.4/5.4 MB[0m [31m60.9 MB/s[0m eta [36m0:00:00[0m:00:01[0m
    [?25hDownloading tensorflow_hub-0.12.0-py2.py3-none-any.whl (108 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m108.8/108.8 kB[0m [31m3.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading tensorflow_model_optimization-0.8.0-py2.py3-none-any.whl (242 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m242.5/242.5 kB[0m [31m7.1 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading absl_py-1.4.0-py3-none-any.whl (126 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m126.5/126.5 kB[0m [31m3.9 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading tensorflowjs-3.18.0-py3-none-any.whl (77 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m77.5/77.5 kB[0m [31m1.9 MB/s[0m eta [36m0:00:00[0meta [36m0:00:01[0m
    [?25hDownloading tflite_support-0.4.3-cp38-cp38-manylinux2014_x86_64.whl (60.8 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m60.8/60.8 MB[0m [31m26.5 MB/s[0m eta [36m0:00:00[0m:00:01[0m00:01[0m
    [?25hDownloading urllib3-1.25.11-py2.py3-none-any.whl (127 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m128.0/128.0 kB[0m [31m4.0 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading astunparse-1.6.3-py2.py3-none-any.whl (12 kB)
    Downloading audioread-3.0.1-py3-none-any.whl (23 kB)
    Downloading cycler-0.12.1-py3-none-any.whl (8.3 kB)
    Downloading dm_tree-0.1.8-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (152 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m152.9/152.9 kB[0m [31m4.6 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading etils-1.3.0-py3-none-any.whl (126 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m126.4/126.4 kB[0m [31m3.7 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading gast-0.5.4-py3-none-any.whl (19 kB)
    Downloading google_api_python_client-2.133.0-py2.py3-none-any.whl (11.8 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m11.8/11.8 MB[0m [31m63.1 MB/s[0m eta [36m0:00:00[0m:00:01[0m0:01[0m
    [?25hDownloading google_cloud_bigquery-3.24.0-py2.py3-none-any.whl (238 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m238.5/238.5 kB[0m [31m7.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading google_pasta-0.2.0-py3-none-any.whl (57 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m57.5/57.5 kB[0m [31m1.6 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading grpcio-1.64.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (5.6 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.6/5.6 MB[0m [31m62.4 MB/s[0m eta [36m0:00:00[0m:00:01[0m
    [?25hDownloading h5py-3.11.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (5.3 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.3/5.3 MB[0m [31m57.8 MB/s[0m eta [36m0:00:00[0m:00:01[0m
    [?25hDownloading joblib-1.4.2-py3-none-any.whl (301 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m301.8/301.8 kB[0m [31m9.0 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading keras-2.8.0-py2.py3-none-any.whl (1.4 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.4/1.4 MB[0m [31m29.5 MB/s[0m eta [36m0:00:00[0m:00:01[0m
    [?25hDownloading Keras_Preprocessing-1.1.2-py2.py3-none-any.whl (42 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m42.6/42.6 kB[0m [31m1.2 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading kiwisolver-1.4.5-cp38-cp38-manylinux_2_5_x86_64.manylinux1_x86_64.whl (1.2 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.2/1.2 MB[0m [31m26.8 MB/s[0m eta [36m0:00:00[0m:00:01[0m
    [?25hDownloading libclang-18.1.1-py2.py3-none-manylinux2010_x86_64.whl (24.5 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m24.5/24.5 MB[0m [31m46.4 MB/s[0m eta [36m0:00:00[0m:00:01[0m00:01[0m
    [?25hDownloading llvmlite-0.41.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (43.6 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m43.6/43.6 MB[0m [31m33.5 MB/s[0m eta [36m0:00:00[0m:00:01[0m00:01[0m
    [?25hDownloading opt_einsum-3.3.0-py3-none-any.whl (65 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m65.5/65.5 kB[0m [31m2.0 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading packaging-20.9-py2.py3-none-any.whl (40 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m40.9/40.9 kB[0m [31m795.4 kB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hDownloading pandas-2.0.3-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (12.4 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m12.4/12.4 MB[0m [31m65.0 MB/s[0m eta [36m0:00:00[0m:00:01[0m00:01[0m
    [?25hDownloading pooch-1.8.2-py3-none-any.whl (64 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m64.6/64.6 kB[0m [31m1.9 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading protobuf-3.19.6-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.1/1.1 MB[0m [31m24.3 MB/s[0m eta [36m0:00:00[0m:00:01[0m
    [?25hDownloading py_cpuinfo-9.0.0-py3-none-any.whl (22 kB)
    Downloading pybind11-2.12.0-py3-none-any.whl (234 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m235.0/235.0 kB[0m [31m7.6 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading pyparsing-3.1.2-py3-none-any.whl (103 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m103.2/103.2 kB[0m [31m3.0 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading requests-2.32.3-py3-none-any.whl (64 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m64.9/64.9 kB[0m [31m2.0 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading resampy-0.4.3-py3-none-any.whl (3.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.1/3.1 MB[0m [31m46.8 MB/s[0m eta [36m0:00:00[0m:00:01[0m
    [?25hDownloading scikit_learn-1.3.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (11.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m11.1/11.1 MB[0m [31m71.2 MB/s[0m eta [36m0:00:00[0m:00:01[0m0:01[0m
    [?25hDownloading scipy-1.10.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (34.5 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m34.5/34.5 MB[0m [31m39.3 MB/s[0m eta [36m0:00:00[0m:00:01[0m00:01[0m
    [?25hDownloading sounddevice-0.4.7-py3-none-any.whl (32 kB)
    Downloading soundfile-0.12.1-py2.py3-none-manylinux_2_31_x86_64.whl (1.2 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.2/1.2 MB[0m [31m26.0 MB/s[0m eta [36m0:00:00[0m:00:01[0m
    [?25hDownloading tensorboard-2.8.0-py3-none-any.whl (5.8 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.8/5.8 MB[0m [31m58.3 MB/s[0m eta [36m0:00:00[0m:00:01[0m00:01[0m
    [?25hDownloading tensorflow_estimator-2.8.0-py2.py3-none-any.whl (462 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m462.3/462.3 kB[0m [31m13.0 MB/s[0m eta [36m0:00:00[0m00:01[0m
    [?25hDownloading tensorflow_io_gcs_filesystem-0.34.0-cp38-cp38-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (2.4 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.4/2.4 MB[0m [31m42.9 MB/s[0m eta [36m0:00:00[0m:00:01[0m
    [?25hDownloading termcolor-2.4.0-py3-none-any.whl (7.7 kB)
    Downloading tf_slim-1.1.0-py2.py3-none-any.whl (352 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m352.1/352.1 kB[0m [31m9.7 MB/s[0m eta [36m0:00:00[0m:00:01[0m
    [?25hDownloading typeguard-2.13.3-py3-none-any.whl (17 kB)
    Downloading wrapt-1.16.0-cp38-cp38-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (83 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m83.4/83.4 kB[0m [31m2.6 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading array_record-0.4.0-py38-none-any.whl (3.0 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.0/3.0 MB[0m [31m42.6 MB/s[0m eta [36m0:00:00[0m:00:01[0m
    [?25hDownloading attrs-23.2.0-py3-none-any.whl (60 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m60.8/60.8 kB[0m [31m1.6 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading click-8.1.7-py3-none-any.whl (97 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m97.9/97.9 kB[0m [31m3.1 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading dataclasses-0.6-py3-none-any.whl (14 kB)
    Downloading gin_config-0.5.0-py3-none-any.whl (61 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m61.3/61.3 kB[0m [31m1.7 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading importlib_resources-6.4.0-py3-none-any.whl (38 kB)
    Downloading opencv_python_headless-4.10.0.82-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (49.9 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m49.9/49.9 MB[0m [31m30.6 MB/s[0m eta [36m0:00:00[0m:00:01[0m00:01[0m
    [?25hDownloading tensorflow_metadata-1.13.0-py3-none-any.whl (53 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m53.3/53.3 kB[0m [31m1.6 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading toml-0.10.2-py2.py3-none-any.whl (16 kB)
    Using cached tqdm-4.66.4-py3-none-any.whl (78 kB)
    Downloading certifi-2024.6.2-py3-none-any.whl (164 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m164.4/164.4 kB[0m [31m5.0 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading cffi-1.16.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (444 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m444.7/444.7 kB[0m [31m12.3 MB/s[0m eta [36m0:00:00[0m00:01[0m
    [?25hDownloading charset_normalizer-3.3.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (141 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m141.1/141.1 kB[0m [31m3.7 MB/s[0m eta [36m0:00:00[0mta [36m0:00:01[0m
    [?25hDownloading google_api_core-2.19.0-py3-none-any.whl (139 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m139.0/139.0 kB[0m [31m4.4 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading google_auth-2.30.0-py2.py3-none-any.whl (193 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m193.7/193.7 kB[0m [31m6.2 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading google_auth_httplib2-0.2.0-py2.py3-none-any.whl (9.3 kB)
    Downloading google_auth_oauthlib-0.4.6-py2.py3-none-any.whl (18 kB)
    Downloading google_cloud_core-2.4.1-py2.py3-none-any.whl (29 kB)
    Downloading google_resumable_media-2.7.1-py2.py3-none-any.whl (81 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m81.2/81.2 kB[0m [31m2.6 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading googleapis_common_protos-1.63.1-py2.py3-none-any.whl (229 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m229.2/229.2 kB[0m [31m7.4 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading httplib2-0.22.0-py3-none-any.whl (96 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m96.9/96.9 kB[0m [31m2.7 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading idna-3.7-py3-none-any.whl (66 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m66.8/66.8 kB[0m [31m2.1 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading Markdown-3.6-py3-none-any.whl (105 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m105.4/105.4 kB[0m [31m3.4 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading pytz-2024.1-py2.py3-none-any.whl (505 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m505.5/505.5 kB[0m [31m13.7 MB/s[0m eta [36m0:00:00[0m00:01[0m
    [?25hDownloading tensorboard_data_server-0.6.1-py3-none-manylinux2010_x86_64.whl (4.9 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m4.9/4.9 MB[0m [31m60.2 MB/s[0m eta [36m0:00:00[0m:00:01[0m
    [?25hDownloading tensorboard_plugin_wit-1.8.1-py3-none-any.whl (781 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m781.3/781.3 kB[0m [31m20.4 MB/s[0m eta [36m0:00:00[0m00:01[0m
    [?25hDownloading threadpoolctl-3.5.0-py3-none-any.whl (18 kB)
    Downloading tzdata-2024.1-py2.py3-none-any.whl (345 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m345.4/345.4 kB[0m [31m10.4 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading uritemplate-4.1.1-py2.py3-none-any.whl (10 kB)
    Downloading werkzeug-3.0.3-py3-none-any.whl (227 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m227.3/227.3 kB[0m [31m7.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading bleach-6.1.0-py3-none-any.whl (162 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m162.8/162.8 kB[0m [31m5.4 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading python_slugify-8.0.4-py2.py3-none-any.whl (10 kB)
    Downloading cachetools-5.3.3-py3-none-any.whl (9.3 kB)
    Downloading google_crc32c-1.5.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (32 kB)
    Downloading grpcio_status-1.48.2-py3-none-any.whl (14 kB)
    Downloading MarkupSafe-2.1.5-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (26 kB)
    Downloading proto_plus-1.23.0-py3-none-any.whl (48 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m48.8/48.8 kB[0m [31m1.4 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading pyasn1_modules-0.4.0-py3-none-any.whl (181 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m181.2/181.2 kB[0m [31m5.2 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading requests_oauthlib-2.0.0-py2.py3-none-any.whl (24 kB)
    Downloading rsa-4.9-py3-none-any.whl (34 kB)
    Downloading text_unidecode-1.3-py2.py3-none-any.whl (78 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m78.2/78.2 kB[0m [31m2.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading pycparser-2.22-py3-none-any.whl (117 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m117.6/117.6 kB[0m [31m3.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading webencodings-0.5.1-py2.py3-none-any.whl (11 kB)
    Downloading oauthlib-3.2.2-py3-none-any.whl (151 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m151.7/151.7 kB[0m [31m4.6 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading pyasn1-0.6.0-py2.py3-none-any.whl (85 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m85.3/85.3 kB[0m [31m2.5 MB/s[0m eta [36m0:00:00[0m
    [?25hBuilding wheels for collected packages: fire, kaggle, promise
      Building wheel for fire (setup.py) ... [?25ldone
    [?25h  Created wheel for fire: filename=fire-0.6.0-py2.py3-none-any.whl size=117029 sha256=d3eaa01b896f4c3d699998ef9bda8a42993779b834d44e0d39bb6fee21d833a3
      Stored in directory: /home/codespace/.cache/pip/wheels/f6/76/a0/afe23f6f3bc186630845efab23bc7a6e348204102070fdf465
      Building wheel for kaggle (setup.py) ... [?25ldone
    [?25h  Created wheel for kaggle: filename=kaggle-1.6.14-py3-none-any.whl size=105119 sha256=1bc98ab1650a345377d458df0f5c6dbf05617ede4d7aae15113accd41d3a0cfb
      Stored in directory: /home/codespace/.cache/pip/wheels/d8/3e/73/d260c9da20f77c8b16dd363273733109776f728fe8af6017a3
      Building wheel for promise (setup.py) ... [?25ldone
    [?25h  Created wheel for promise: filename=promise-2.3-py3-none-any.whl size=21483 sha256=882c55781c097cef423bb928991ef9e0bd36efbb97755bb4a6a13b4ae961dbe8
      Stored in directory: /home/codespace/.cache/pip/wheels/54/aa/01/724885182f93150035a2a91bce34a12877e8067a97baaf5dc8
    Successfully built fire kaggle promise
    Installing collected packages: webencodings, text-unidecode, tensorflow-estimator, tensorboard-plugin-wit, sentencepiece, pytz, py-cpuinfo, libclang, keras, gin-config, flatbuffers, dm-tree, dataclasses, wrapt, urllib3, uritemplate, tzdata, typeguard, tqdm, toml, threadpoolctl, termcolor, tensorflow-io-gcs-filesystem, tensorboard-data-server, PyYAML, python-slugify, pyparsing, pycparser, pybind11, pyasn1, protobuf, promise, pillow, oauthlib, numpy, MarkupSafe, lxml, llvmlite, kiwisolver, joblib, importlib-resources, idna, grpcio, google-pasta, google-crc32c, gast, etils, Cython, cycler, click, charset-normalizer, certifi, cachetools, bleach, audioread, attrs, astunparse, absl-py, werkzeug, tf-slim, tensorflow-model-optimization, tensorflow-hub, scipy, rsa, requests, pyasn1-modules, proto-plus, pandas, packaging, opt-einsum, opencv-python-headless, numba, matplotlib, markdown, keras-preprocessing, httplib2, h5py, googleapis-common-protos, google-resumable-media, fire, CFFI, tensorflow-metadata, tensorflow-addons, soundfile, sounddevice, scikit-learn, resampy, requests-oauthlib, pooch, neural-structured-learning, kaggle, grpcio-status, google-auth, tflite-support, librosa, google-auth-oauthlib, google-auth-httplib2, google-api-core, array-record, tensorflow-datasets, tensorboard, google-cloud-core, google-api-python-client, tensorflow, google-cloud-bigquery, tf-models-official, tensorflowjs, scann, tflite-model-maker
      Attempting uninstall: packaging
        Found existing installation: packaging 24.1
        Uninstalling packaging-24.1:
          Successfully uninstalled packaging-24.1
    Successfully installed CFFI-1.16.0 Cython-3.0.10 MarkupSafe-2.1.5 PyYAML-6.0.1 absl-py-1.4.0 array-record-0.4.0 astunparse-1.6.3 attrs-23.2.0 audioread-3.0.1 bleach-6.1.0 cachetools-5.3.3 certifi-2024.6.2 charset-normalizer-3.3.2 click-8.1.7 cycler-0.12.1 dataclasses-0.6 dm-tree-0.1.8 etils-1.3.0 fire-0.6.0 flatbuffers-24.3.25 gast-0.5.4 gin-config-0.5.0 google-api-core-2.19.0 google-api-python-client-2.133.0 google-auth-2.30.0 google-auth-httplib2-0.2.0 google-auth-oauthlib-0.4.6 google-cloud-bigquery-3.24.0 google-cloud-core-2.4.1 google-crc32c-1.5.0 google-pasta-0.2.0 google-resumable-media-2.7.1 googleapis-common-protos-1.63.1 grpcio-1.64.1 grpcio-status-1.48.2 h5py-3.11.0 httplib2-0.22.0 idna-3.7 importlib-resources-6.4.0 joblib-1.4.2 kaggle-1.6.14 keras-2.8.0 keras-preprocessing-1.1.2 kiwisolver-1.4.5 libclang-18.1.1 librosa-0.8.1 llvmlite-0.41.1 lxml-5.2.2 markdown-3.6 matplotlib-3.4.3 neural-structured-learning-1.4.0 numba-0.58.1 numpy-1.23.3 oauthlib-3.2.2 opencv-python-headless-4.10.0.82 opt-einsum-3.3.0 packaging-20.9 pandas-2.0.3 pillow-10.3.0 pooch-1.8.2 promise-2.3 proto-plus-1.23.0 protobuf-3.19.6 py-cpuinfo-9.0.0 pyasn1-0.6.0 pyasn1-modules-0.4.0 pybind11-2.12.0 pycparser-2.22 pyparsing-3.1.2 python-slugify-8.0.4 pytz-2024.1 requests-2.32.3 requests-oauthlib-2.0.0 resampy-0.4.3 rsa-4.9 scann-1.2.6 scikit-learn-1.3.2 scipy-1.10.1 sentencepiece-0.2.0 sounddevice-0.4.7 soundfile-0.12.1 tensorboard-2.8.0 tensorboard-data-server-0.6.1 tensorboard-plugin-wit-1.8.1 tensorflow-2.8.4 tensorflow-addons-0.21.0 tensorflow-datasets-4.9.0 tensorflow-estimator-2.8.0 tensorflow-hub-0.12.0 tensorflow-io-gcs-filesystem-0.34.0 tensorflow-metadata-1.13.0 tensorflow-model-optimization-0.8.0 tensorflowjs-3.18.0 termcolor-2.4.0 text-unidecode-1.3 tf-models-official-2.3.0 tf-slim-1.1.0 tflite-model-maker-0.4.3 tflite-support-0.4.3 threadpoolctl-3.5.0 toml-0.10.2 tqdm-4.66.4 typeguard-2.13.3 tzdata-2024.1 uritemplate-4.1.1 urllib3-1.25.11 webencodings-0.5.1 werkzeug-3.0.3 wrapt-1.16.0

### 在本地缓存中更新可用包的列表

```python
!sudo apt-get update
```

    Get:1 http://security.ubuntu.com/ubuntu focal-security InRelease [128 kB]
    Get:2 https://packages.microsoft.com/repos/microsoft-ubuntu-focal-prod focal InRelease [3632 B]
    Get:3 https://dl.yarnpkg.com/debian stable InRelease [17.1 kB]                 
    Get:5 http://archive.ubuntu.com/ubuntu focal InRelease [265 kB]                
    Get:6 https://packages.microsoft.com/repos/microsoft-ubuntu-focal-prod focal/main all Packages [2714 B]
    Get:7 https://packages.microsoft.com/repos/microsoft-ubuntu-focal-prod focal/main amd64 Packages [294 kB]
    Get:8 https://repo.anaconda.com/pkgs/misc/debrepo/conda stable InRelease [3961 B]
    Get:9 https://dl.yarnpkg.com/debian stable/main amd64 Packages [11.8 kB]       
    Get:10 https://dl.yarnpkg.com/debian stable/main all Packages [11.8 kB]        
    Get:4 https://packagecloud.io/github/git-lfs/ubuntu focal InRelease [28.0 kB]  
    Get:11 http://security.ubuntu.com/ubuntu focal-security/universe amd64 Packages [1213 kB]
    Get:12 https://repo.anaconda.com/pkgs/misc/debrepo/conda stable/main amd64 Packages [4557 B]
    Get:13 https://packagecloud.io/github/git-lfs/ubuntu focal/main amd64 Packages [3690 B]
    Get:14 http://security.ubuntu.com/ubuntu focal-security/multiverse amd64 Packages [29.8 kB]
    Get:15 http://security.ubuntu.com/ubuntu focal-security/restricted amd64 Packages [3651 kB]
    Get:16 http://archive.ubuntu.com/ubuntu focal-updates InRelease [128 kB]       
    Get:17 http://security.ubuntu.com/ubuntu focal-security/main amd64 Packages [3710 kB]
    Get:18 http://archive.ubuntu.com/ubuntu focal-backports InRelease [128 kB]     
    Get:19 http://archive.ubuntu.com/ubuntu focal/restricted amd64 Packages [33.4 kB]
    Get:20 http://archive.ubuntu.com/ubuntu focal/multiverse amd64 Packages [177 kB]
    Get:21 http://archive.ubuntu.com/ubuntu focal/universe amd64 Packages [11.3 MB]
    Get:22 http://archive.ubuntu.com/ubuntu focal/main amd64 Packages [1275 kB]
    Get:23 http://archive.ubuntu.com/ubuntu focal-updates/multiverse amd64 Packages [32.5 kB]
    Get:24 http://archive.ubuntu.com/ubuntu focal-updates/universe amd64 Packages [1511 kB]
    Get:25 http://archive.ubuntu.com/ubuntu focal-updates/restricted amd64 Packages [3800 kB]
    Get:26 http://archive.ubuntu.com/ubuntu focal-updates/main amd64 Packages [4182 kB]
    Get:27 http://archive.ubuntu.com/ubuntu focal-backports/main amd64 Packages [55.2 kB]
    Get:28 http://archive.ubuntu.com/ubuntu focal-backports/universe amd64 Packages [28.6 kB]
    Fetched 32.1 MB in 3s (9318 kB/s)                             
    Reading package lists... Done

### 安装 `libusb-1.0-0` 库

```python
!sudo apt-get install libusb-1.0-0
```

    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    The following NEW packages will be installed:
      libusb-1.0-0
    0 upgraded, 1 newly installed, 0 to remove and 21 not upgraded.
    Need to get 46.5 kB of archives.
    After this operation, 139 kB of additional disk space will be used.
    Get:1 http://archive.ubuntu.com/ubuntu focal/main amd64 libusb-1.0-0 amd64 2:1.0.23-2build1 [46.5 kB]
    Fetched 46.5 kB in 0s (149 kB/s)        
    Selecting previously unselected package libusb-1.0-0:amd64.
    (Reading database ... 70090 files and directories currently installed.)
    Preparing to unpack .../libusb-1.0-0_2%3a1.0.23-2build1_amd64.deb ...
    Unpacking libusb-1.0-0:amd64 (2:1.0.23-2build1) ...
    Setting up libusb-1.0-0:amd64 (2:1.0.23-2build1) ...
    Processing triggers for libc-bin (2.31-0ubuntu9.16) ...

### 导入相关的库


```python
import os

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt

```

    2024-06-14 14:13:58.061846: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
    2024-06-14 14:13:58.061881: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
    /workspaces/codespaces-jupyter/.conda/lib/python3.8/site-packages/tensorflow_addons/utils/tfa_eol_msg.py:23: UserWarning: 
    
    TensorFlow Addons (TFA) has ended development and introduction of new features.
    TFA has entered a minimal maintenance and release mode until a planned end of life in May 2024.
    Please modify downstream libraries to take dependencies from other repositories in our TensorFlow community (e.g. Keras, Keras-CV, and Keras-NLP). 
    
    For more information see: https://github.com/tensorflow/addons/issues/2807 
    
      warnings.warn(
    /workspaces/codespaces-jupyter/.conda/lib/python3.8/site-packages/tensorflow_addons/utils/ensure_tf_install.py:53: UserWarning: Tensorflow Addons supports using Python ops for all Tensorflow versions above or equal to 2.11.0 and strictly below 2.14.0 (nightly versions are not supported). 
     The versions of TensorFlow you are currently using is 2.8.4 and is not supported. 
    Some things might work, some things might not.
    If you were to encounter a bug, do not file an issue.
    If you want to make sure you're using a tested and supported configuration, either change the TensorFlow version or the TensorFlow Addons's version. 
    You can find the compatibility matrix in TensorFlow Addon's readme:
    https://github.com/tensorflow/addons
      warnings.warn(
    /workspaces/codespaces-jupyter/.conda/lib/python3.8/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm

## 模型训练

### 获取数据

先从较小的数据集开始训练。

```python
image_path = tf.keras.utils.get_file(
      'flower_photos.tgz',
      'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      extract=True)
image_path = os.path.join(os.path.dirname(image_path), 'flower_photos')

```

    Downloading data from https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    228818944/228813984 [==============================] - 7s 0us/step
    228827136/228813984 [==============================] - 7s 0us/step

## 运行示例

### 第一步：加载数据集，并将数据集分为训练数据和测试数据。

```python
data = DataLoader.from_folder(image_path)
train_data, test_data = data.split(0.9)

```

    INFO:tensorflow:Load image with size: 3670, num_label: 5, labels: daisy, dandelion, roses, sunflowers, tulips.


    2024-06-14 14:16:30.966806: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /workspaces/codespaces-jupyter/.conda/lib/python3.8/site-packages/cv2/../../lib64:
    2024-06-14 14:16:30.966835: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
    2024-06-14 14:16:30.966855: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (codespaces-11eef8): /proc/driver/nvidia/version does not exist
    2024-06-14 14:16:30.967112: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.

### 第二步：训练Tensorflow模型

```python
model = image_classifier.create(train_data)

```

    INFO:tensorflow:Retraining the models...
    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     hub_keras_layer_v1v2 (HubKe  (None, 1280)             3413024   
     rasLayerV1V2)                                                   
                                                                     
     dropout (Dropout)           (None, 1280)              0         
                                                                     
     dense (Dense)               (None, 5)                 6405      
                                                                     
    =================================================================
    Total params: 3,419,429
    Trainable params: 6,405
    Non-trainable params: 3,413,024
    _________________________________________________________________
    None
    Epoch 1/5


    2024-06-14 14:16:53.917737: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 51380224 exceeds 10% of free system memory.
    2024-06-14 14:16:54.031990: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 51380224 exceeds 10% of free system memory.
    2024-06-14 14:16:54.084009: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 51380224 exceeds 10% of free system memory.
    2024-06-14 14:16:54.105070: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 154140672 exceeds 10% of free system memory.
    2024-06-14 14:16:54.141250: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 38535168 exceeds 10% of free system memory.


    103/103 [==============================] - 46s 431ms/step - loss: 0.8631 - accuracy: 0.7743
    Epoch 2/5
    103/103 [==============================] - 45s 435ms/step - loss: 0.6580 - accuracy: 0.8914
    Epoch 3/5
    103/103 [==============================] - 43s 414ms/step - loss: 0.6173 - accuracy: 0.9129
    Epoch 4/5
    103/103 [==============================] - 46s 442ms/step - loss: 0.5985 - accuracy: 0.9254
    Epoch 5/5
    103/103 [==============================] - 44s 422ms/step - loss: 0.5898 - accuracy: 0.9293

### 第三步：评估模型

```python
loss, accuracy = model.evaluate(test_data)

```

    12/12 [==============================] - 6s 384ms/step - loss: 0.6162 - accuracy: 0.9019

### 第四步，导出Tensorflow Lite模型

```python
model.export(export_dir='.')

```

    2024-06-14 14:22:46.922097: W tensorflow/python/util/util.cc:368] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.


    INFO:tensorflow:Assets written to: /tmp/tmpjoko9w6i/assets


    INFO:tensorflow:Assets written to: /tmp/tmpjoko9w6i/assets
    2024-06-14 14:22:50.410958: I tensorflow/core/grappler/devices.cc:66] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 0
    2024-06-14 14:22:50.411082: I tensorflow/core/grappler/clusters/single_machine.cc:358] Starting new session
    2024-06-14 14:22:50.437719: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:1164] Optimization results for grappler item: graph_to_optimize
      function_optimizer: Graph size after: 913 nodes (656), 923 edges (664), time = 14.915ms.
      function_optimizer: function_optimizer did nothing. time = 0.006ms.
    
    /workspaces/codespaces-jupyter/.conda/lib/python3.8/site-packages/tensorflow/lite/python/convert.py:746: UserWarning: Statistics for quantized inputs were expected, but not specified; continuing anyway.
      warnings.warn("Statistics for quantized inputs were expected, but not "
    2024-06-14 14:22:50.989015: W tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.cc:357] Ignored output_format.
    2024-06-14 14:22:50.989061: W tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.cc:360] Ignored drop_control_dependency.


    INFO:tensorflow:Label file is inside the TFLite model with metadata.


    fully_quantize: 0, inference_type: 6, input_inference_type: 3, output_inference_type: 3
    INFO:tensorflow:Label file is inside the TFLite model with metadata.


    INFO:tensorflow:Saving labels in /tmp/tmpq1wll9li/labels.txt


    INFO:tensorflow:Saving labels in /tmp/tmpq1wll9li/labels.txt


    INFO:tensorflow:TensorFlow Lite model exported successfully: ./model.tflite


    INFO:tensorflow:TensorFlow Lite model exported successfully: ./model.tflite



```python

```
