# Polish Medical ASR – Fine-tuned Whisper

**Fine-tuned Whisper model for recognizing Polish drug names in medical interviews.**

---

## Base Model
[Whisper Medium Medical PL](https://www.kaggle.com/models/msxksm/whisper-medium-medical-pl)  
This model was previously fine-tuned on a Polish medical corpus.

---

## Overview
The goal of this project is to develop a Minimum Viable Product (MVP) of an Automatic Speech Recognition (ASR) system capable of recognizing the most common Polish drug names during medical interviews.  
This project was carried out for the [**Medwave**](https://www.linkedin.com/company/aimedwave/?originalSubdomain=pl) initiative, which aims to make doctor–patient interactions more patient-centered.

---

## Features
- Recognizes more drug names than the base model (improved average wer).  
- Provides scripts to calculate **word-level confidence scores**.  
- Supports adding **custom context prompts** to improve recognition accuracy.  

---

## Model Details
| Property | Value |
|----------|-------|
| Model type | `WhisperForConditionalGeneration` |
| Pretrained from | `whisper-medium-medical-pl` |
| Fine-tuned on | Synthetic data with Polish drug names |
| License | MIT |
| Hugging Face Hub | [medical-polish-drugs-whisper](https://huggingface.co/pwysoc/medical-polish-drugs-whisper) |

---

## Dataset
The dataset was **synthetically generated**:  
1. Typical sentences from medical interviews were generated, including common Polish drug names.  
2. Audio was synthesized using three voices (1 female, 2 male) via **Rhasspy TTS models**.  
3. To improve robustness, the audio was augmented with **reverberation** and **hospital background noise**.

> **Note:** Real patient recordings are planned for future iterations to improve recognition quality.

**Challenge:** Some drug names were generated automatically, which led to ambiguous cases in pronunciation.  
For example, in Polish the letter "c" can be pronounced in two ways: as "c" or as "k" (this is the case for drug names like Bibloc).

---

## Training Details
Due to limited computational resources (CPU-only, no GPU), only **basic fine-tuning** was conducted.  
We used **LoRA** to adapt the pretrained model for a very specific drug-name recognition task.  
A **weighted loss** was applied: errors on drug names were penalized **10x more** than regular words.

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 1 |
| Batch size | 2 |
| Learning rate | 9e-4 |
| LoRA rank (r) | 10 |
| Optimizer | AdamW |
| Training time | ~12h (including evaluation every 20 steps) |

> For full details, see `Fine-tuning.ipynb`.

---

## Evaluation
- Metrics: **WER**, **CER**  
- Confidence analysis on individual words  
- Drug recognition metrics: **precision**, **recall**  
- Testing on **real human voices**

---

## Usage
To use this model, you need a WAV file with a **16 kHz sampling rate**.  

```python
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

language = "pl"
task = "transcribe"

processor = WhisperProcessor.from_pretrained("pwysoc/medical-polish-drugs-whisper")
model = WhisperForConditionalGeneration.from_pretrained("pwysoc/medical-polish-drugs-whisper")

waveform, sr = torchaudio.load("file_with_polish_drug.wav")
SAMPLING_RATE = 16000

if sr != SAMPLING_RATE:
    waveform = torchaudio.transforms.Resample(sr, SAMPLING_RATE)(waveform)

if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)

# Extract features
input_features = processor(
    waveform.squeeze(0),
    sampling_rate=SAMPLING_RATE,
    return_tensors="pt"
).input_features

# This parameter can be adjusted to add more context on recognition (if used properly, it improves results a lot!)
initial_prompt = (
    "To nagranie jest fragmentem z wywiadu medycznego. "
    "Zawiera nazwy leków takie jak Paracetamol, Ibuprom."
)
decoder_input_ids = None

prompt_ids = processor.tokenizer( initial_prompt, add_special_tokens=False, return_tensors="pt" ).input_ids 
forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
forced_ids = torch.tensor([[tok_id for _, tok_id in forced_decoder_ids]]) 
decoder_input_ids = torch.cat([prompt_ids, forced_ids], dim=1) 
predicted_ids = model.generate( input_features, decoder_input_ids=decoder_input_ids)[0] 
transcription = processor.decode(predicted_ids, skip_special_tokens=True) 
print(transcription)
```
## Further Work

- Use `word_confidence.txt` to identify drug names that require more training data.
- Enrich the dataset with real recordings to improve robustness.
- Explore hyperparameter optimization for better WER and CER metrics.
- Integrate with medical interview systems for real-time transcription.
- Double-check the correctness of ambiguous drug name pronunciations.

## Suggested Improvements / Future Directions

- Evaluate model performance on longer multi-drug sentences.
- Explore domain-specific vocabularies for other medical entities (e.g., procedures, symptoms).
- Perform continuous evaluation and data enrichment using word-level confidence scores.
- Experiment with different prompt strategies, context injection and usage of banned words for improved drug recognition.

