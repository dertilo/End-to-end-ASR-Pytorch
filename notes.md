# setup
* on gunther: `pip install torch==1.2.0 torchaudio==0.3.0`

python3 util/generate_vocab_file.py --input_file TEXT_FILE --output_file OUTPUT_FILE #TODO(tilo): what for?

python main.py

tensorboard --logdir data/tmp/log --bind_all

### train language-model
1. unzip file: `gunzip librispeech-lm-norm.txt.gz`
2. train on second GPU`CUDA_VISIBLE_DEVICES=1 python3 main.py --config config/libri/lm_example.yaml --lm`

## data
* [language-model-data](http://www.openslr.org/11/) `wget --trust-server-names http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz`

### [libi-speech](http://www.openslr.org/12/)

#### ASR libri-speech english
    (base) root@gunther:~/SPEECH/End-to-end-ASR-Pytorch# python main.py
    [INFO] Exp. name : asr_example_sd0
    [INFO] Loading data... large corpus may took a while.
    [INFO] Data spec. | Corpus = Librispeech (from /docker-share/data/libri-speech)
    [INFO]            | Train sets = ['train-clean-100']    | Number of utts = 28539
    [INFO]            | Dev sets = ['dev-clean']    | Number of utts = 2703
    [INFO]            | Batch size = 8              | Bucketing = True
    [INFO] I/O spec.  | Audio feature = fbank       | feature dim = 120     | Token type = subword  | Vocab size = 16000
    [INFO] Model spec.| Encoder's downsampling rate of time axis is 4.
    [INFO]            | VGG Extractor w/ time downsampling rate = 4 in encoder enabled.
    [INFO]            | loc attention decoder enabled ( lambda = 1.0).
    [INFO] Optim.spec.| Algo. = Adadelta    | Lr = 1.0       (Scheduler = fixed)| Scheduled sampling = False
    [INFO] Total training steps 1.0M.
    [INFO] Saved checkpoint (step = 1.0 , wer = 1.20) and status @ ckpt/asr_example_sd0/best_att.pth
    [INFO] Saved checkpoint (step = 5.0K, wer = 0.99) and status @ ckpt/asr_example_sd0/best_att.pth
    [INFO] Saved checkpoint (step = 510.0K, wer = 0.27) and status @ ckpt/asr_example_sd0/best_att.pth
