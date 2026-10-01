# Decision fine-tuning feasibility pilot

`decision_feasibility.py` runs a short full-model fine-tuning trial of the saved
30M scalar and channel-wise TinyStories checkpoints. It uses `choice` examples
from the **training split only** of `LocalLLaMA/typed-decisions`. It makes a
reproducible 80/20 internal split and limits the first run to 48 training and
16 validation cases. The benchmark test split is never loaded.

The pilot adds a fixed-size choice head for one question in the
`customer_service` workflow. It includes the state, question, and option
descriptions in the prompt, and trains against the provided soft probability
targets. It is a feasibility test, not the final dynamic-option model.

## Colab T4

Use a GPU runtime. Install only the data/tokenizer dependencies; Colab already
ships PyTorch. Its current Python version may not satisfy this repository's
Python 3.11 project pin.

```python
%pip install datasets==4.6.1 tiktoken==0.12.0 gdown
```

In the repository directory, upload `decision_feasibility.py` from the local
working tree if it has not been pushed to GitHub:

```python
from google.colab import files
%cd /content/Major-Project-Research
files.upload()  # choose decision_feasibility.py if absent from the clone
```

The supplied Drive link contains `results.tar.gz`. This command downloads it
and streams out only the scalar seed 43 and channel-wise seed 42 checkpoint
files. It accepts `output/tinystories/...`, `outputs/tinystories/...`, or an
additional enclosing directory inside the archive:

```python
!python decision_feasibility.py \
  --variant both \
  --checkpoint-archive-url \
  --steps 5 --batch-size 2 --max-length 256
```

If the link needs Google authentication or Drive blocks direct download,
mount Drive and pass the file path instead:

```python
from google.colab import drive
drive.mount('/content/drive')
!python decision_feasibility.py --variant both \
  --checkpoint-archive '/content/drive/MyDrive/results.tar.gz' \
  --steps 5 --batch-size 2 --max-length 256
```

Download the report:

```python
files.download('output/decision_feasibility/result.json')
```

The report contains measured step times, first and last training loss,
validation soft-label loss, and GPU peak allocated/reserved memory. A
successful run establishes that **this** configuration fits and trains for a
few steps on the assigned GPU. It does not establish final accuracy, the
runtime of a full study, or the feasibility of longer inputs and larger
batches. If memory is tight, try `--batch-size 1`; record the change when
comparing results.

The selected checkpoint files are about 350 MB each and are training outputs,
not part of a normal Git clone. The archive is downloaded into temporary
storage, and only the two selected regular files are copied out. The script
uses `weights_only=True` when loading them; only use an archive from a trusted
source.
