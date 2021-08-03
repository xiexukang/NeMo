#!/usr/bin/env python
# coding: utf-8


import json
import os
import tempfile
from datetime import datetime
from typing import List

import editdistance
import ipdb
import numpy as np
import torch
import wget
from collections import OrderedDict as od
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torchmetrics import Metric
from tqdm.auto import tqdm

from nemo.collections.asr.models import ClusteringDiarizer, EncDecCTCModel
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels, write_rttm2manifest
from nemo.utils import logging


class WER(Metric):
    def __init__(
        self,
        vocabulary,
        batch_dim_index=0,
        use_cer=False,
        ctc_decode=True,
        log_prediction=True,
        dist_sync_on_step=False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=False)
        self.batch_dim_index = batch_dim_index
        self.blank_id = len(vocabulary)
        self.labels_map = dict([(i, vocabulary[i]) for i in range(len(vocabulary))])
        self.use_cer = use_cer
        self.ctc_decode = ctc_decode
        self.log_prediction = log_prediction

        self.add_state("scores", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)
        self.add_state("words", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)

    def ctc_decoder_predictions_tensor(
        self,
        predictions: torch.Tensor,
        predictions_len: torch.Tensor = None,
        return_hypotheses: bool = False,
        return_timestamps: bool = False,
    ) -> List[str]:
        hypotheses, timestamps = [], []

        # Drop predictions to CPU
        prediction_cpu_tensor = predictions.long().cpu()
        # iterate over batch
        for ind in range(prediction_cpu_tensor.shape[self.batch_dim_index]):
            prediction = prediction_cpu_tensor[ind].detach().numpy().tolist()
            if predictions_len is not None:
                prediction = prediction[: predictions_len[ind]]
            # CTC decoding procedure
            decoded_prediction = []
            decoded_timing_list = []
            previous = self.blank_id
            for pdx, p in enumerate(prediction):
                if (p != previous or previous == self.blank_id) and p != self.blank_id:
                    decoded_prediction.append(p)
                    decoded_timing_list.append(pdx)
                previous = p

            if return_timestamps:
                text, timestamp_list = self.decode_tokens_to_str_with_ts(decoded_prediction, decoded_timing_list)
            else:
                text, timestamp_list = self.decode_tokens_to_str(decoded_prediction, decoded_timing_list), None

            if not return_hypotheses:
                hypothesis = text
            else:
                hypothesis = Hypothesis(
                    y_sequence=None,
                    score=-1.0,
                    text=text,
                    alignments=prediction,
                    length=predictions_len[ind] if predictions_len is not None else 0,
                )

            hypotheses.append(hypothesis)
            timestamps.append(timestamp_list)

        if return_timestamps:
            return hypotheses, timestamps
        else:
            return hypotheses

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        predictions_lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        words = 0.0
        scores = 0.0
        references = []
        with torch.no_grad():
            # prediction_cpu_tensor = tensors[0].long().cpu()
            targets_cpu_tensor = targets.long().cpu()
            tgt_lenths_cpu_tensor = target_lengths.long().cpu()

            # iterate over batch
            for ind in range(targets_cpu_tensor.shape[self.batch_dim_index]):
                tgt_len = tgt_lenths_cpu_tensor[ind].item()
                target = targets_cpu_tensor[ind][:tgt_len].numpy().tolist()
                reference = self.decode_tokens_to_str(target)
                references.append(reference)
            if self.ctc_decode:
                hypotheses = self.ctc_decoder_predictions_tensor(predictions, predictions_lengths)
            else:
                raise NotImplementedError("Implement me if you need non-CTC decode on predictions")

        if self.log_prediction:
            logging.info(f"\n")
            logging.info(f"reference:{references[0]}")
            logging.info(f"predicted:{hypotheses[0]}")

        for h, r in zip(hypotheses, references):
            if self.use_cer:
                h_list = list(h)
                r_list = list(r)
            else:
                h_list = h.split()
                r_list = r.split()
            words += len(r_list)
            # Compute Levenstein's distance
            scores += editdistance.eval(h_list, r_list)

        self.scores = torch.tensor(scores, device=self.scores.device, dtype=self.scores.dtype)
        self.words = torch.tensor(words, device=self.words.device, dtype=self.words.dtype)
        # return torch.tensor([scores, words]).to(predictions.device)

    def decode_tokens_to_str(self, tokens: List[int]) -> str:
        hypothesis = ''.join(self.decode_ids_to_tokens(tokens))
        return hypothesis

    def decode_ids_to_tokens(self, tokens: List[int]) -> List[str]:
        token_list = [self.labels_map[c] for c in tokens if c != self.blank_id]
        return token_list

    def decode_tokens_to_str_with_ts(self, tokens: List[int], timestamps: List[int]) -> str:
        hypothesis_list, timestamp_list = self.decode_ids_to_tokens_with_ts(tokens, timestamps)
        hypothesis = ''.join(self.decode_ids_to_tokens(tokens))
        return hypothesis, timestamp_list

    def decode_ids_to_tokens_with_ts(self, tokens: List[int], timestamps: List[int]) -> List[str]:
        token_list = []
        timestamp_list = []
        for i, c in enumerate(tokens):
            if c != self.blank_id:
                token_list.append(self.labels_map[c])
                timestamp_list.append(timestamps[i])
        # token_list = [self.labels_map[c] for c in tokens if c != self.blank_id]
        return token_list, timestamp_list

    def compute(self):
        scores = self.scores.detach().float()
        words = self.words.detach().float()
        return scores / words, scores, words


# class EncDecCTCModel(ASRModel, ExportableEncDecModel, ASRModuleMixin):
class EncDecCTCModel4Diar(EncDecCTCModel):
    """Base class for encoder decoder CTC-based models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        # Global_rank and local_rank is set by LightningModule in Lightning 1.2.0
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_gpus

        super().__init__(cfg=cfg, trainer=trainer)

        self._wer = WER(
            vocabulary=self.decoder.vocabulary,
            batch_dim_index=0,
            use_cer=self._cfg.get('use_cer', False),
            ctc_decode=True,
            dist_sync_on_step=True,
            log_prediction=self._cfg.get("log_prediction", False),
        )

    @torch.no_grad()
    def transcribe(
        self,
        paths2audio_files: List[str],
        batch_size: int = 4,
        logprobs: bool = False,
        return_hypotheses: bool = False,
        return_text_with_logprobs_and_ts: bool = False,
    ) -> List[str]:

        if paths2audio_files is None or len(paths2audio_files) == 0:
            return {}

        if return_hypotheses and logprobs:
            raise ValueError(
                "Either `return_hypotheses` or `logprobs` can be True at any given time."
                "Returned hypotheses will contain the logprobs."
            )

        # We will store transcriptions here
        hypotheses = []
        # Model's mode and device
        mode = self.training
        device = next(self.parameters()).device
        dither_value = self.preprocessor.featurizer.dither
        pad_to_value = self.preprocessor.featurizer.pad_to

        try:
            self.preprocessor.featurizer.dither = 0.0
            self.preprocessor.featurizer.pad_to = 0
            # Switch model to evaluation mode
            self.eval()
            # Freeze the encoder and decoder modules
            self.encoder.freeze()
            self.decoder.freeze()
            logging_level = logging.get_verbosity()
            logging.set_verbosity(logging.WARNING)
            # Work in tmp directory - will store manifest file there
            with tempfile.TemporaryDirectory() as tmpdir:
                with open(os.path.join(tmpdir, 'manifest.json'), 'w') as fp:
                    for audio_file in paths2audio_files:
                        entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': 'nothing'}
                        fp.write(json.dumps(entry) + '\n')

                config = {'paths2audio_files': paths2audio_files, 'batch_size': batch_size, 'temp_dir': tmpdir}

                temporary_datalayer = self._setup_transcribe_dataloader(config)
                for test_batch in tqdm(temporary_datalayer, desc="Transcribing"):
                    logits, logits_len, greedy_predictions = self.forward(
                        input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device)
                    )
                    for idx in range(logits.shape[0]):
                        lg = logits[idx][: logits_len[idx]]
                    if logprobs:
                        # dump log probs per file
                        hypotheses.append(lg.cpu().numpy())
                    else:
                        decoder_output = self._wer.ctc_decoder_predictions_tensor(
                            greedy_predictions,
                            predictions_len=logits_len,
                            return_hypotheses=return_hypotheses,
                            return_timestamps=return_text_with_logprobs_and_ts,
                        )
                        if return_text_with_logprobs_and_ts:
                            current_hypotheses, timestamps = decoder_output
                            logit_out = lg.cpu().numpy()
                            hypotheses.append([current_hypotheses[0], logit_out, timestamps[0]])
                        else:
                            current_hypotheses = decoder_output
                            if return_hypotheses:
                                # dump log probs per file
                                for idx in range(logits.shape[0]):
                                    current_hypotheses[idx].y_sequence = logits[idx][: logits_len[idx]]

                            hypotheses += current_hypotheses

                    del greedy_predictions
                    del logits
                    del test_batch
        finally:
            # set mode back to its original value
            self.train(mode=mode)
            self.preprocessor.featurizer.dither = dither_value
            self.preprocessor.featurizer.pad_to = pad_to_value
            if mode is True:
                self.encoder.unfreeze()
                self.decoder.unfreeze()
            logging.set_verbosity(logging_level)

        return hypotheses


def _get_silence_timestamps(probs, symbol_idx, state_symbol):
    spaces = []
    idx_state = 0
    state = ''

    if np.argmax(probs[0]) == symbol_idx:
        state = state_symbol

    for idx in range(1, probs.shape[0]):
        current_char_idx = np.argmax(probs[idx])
        if state == state_symbol and current_char_idx != 0 and current_char_idx != symbol_idx:
            spaces.append([idx_state, idx - 1])
            state = ''
        if state == '':
            if current_char_idx == symbol_idx:
                state = state_symbol
                idx_state = idx

    if state == state_symbol:
        spaces.append([idx_state, len(probs) - 1])

    return spaces


def dump_json_to_file(file_path, riva_dict):
    with open(file_path, "w") as outfile:
        json.dump(riva_dict, outfile, indent=4)


def write_txt(w_path, val):
    with open(w_path, "w") as output:
        output.write(val + '\n')
    return None


def write_json_and_transcript(
    ROOT,
    audio_file_list,
    diar_labels,
    word_list,
    word_ts_list,
    params,
):
    total_riva_dict = {}
    for k, audio_file_path in enumerate(audio_file_list):
        uniq_id = get_uniq_id_from_audio_path(audio_file_path)
        labels = diar_labels[k]
        audacity_label_words=[]

        ipdb.set_trace()
        n_spk = get_num_of_spk_from_labels(labels)
        string_out = ''
        riva_dict = od({
            'status': 'Success',
            'session_id': uniq_id,
            'transcription': ' '.join(word_list[k]),
            'speaker_count': n_spk,
            'words': [],
        })

        start_point, end_point, speaker = labels[0].split()

        words = word_list[k]
        logging.info(f"Creating results for Session: {uniq_id} n_spk: {n_spk} ")
        string_out = print_time(string_out, speaker, start_point, end_point, params)

        word_pos, idx = 0, 0
        for j, word_ts_stt_end in enumerate(word_ts_list[k]):

            word_pos = params['offset'] + word_ts_stt_end[0] * params['time_stride']
            if word_pos < float(end_point):
                string_out = print_word(string_out, words[j], params)
            else:
                idx += 1
                idx = min(idx, len(labels)-1)
                start_point, end_point, speaker = labels[idx].split()
                string_out = print_time(string_out, speaker, start_point, end_point, params)
                string_out = print_word(string_out, words[j], params)

            stt_sec, end_sec = get_timestamp_in_sec(word_ts_stt_end, params)
            riva_dict = add_json_to_dict(
                riva_dict, words[j], stt_sec, end_sec, speaker, params
            )  # params['offset'], time_stride, round_float)

            total_riva_dict[uniq_id] = riva_dict
            audacity_label_words = get_audacity_label(words[j], 
                                                      stt_sec, end_sec,
                                                      speaker,
                                                      audacity_label_words)
            # pos_prev = pos_end

        logging.info(f"Writing {ROOT}/{uniq_id}.json")
        dump_json_to_file(f'{ROOT}/{uniq_id}.json', riva_dict)

        logging.info(f"Writing {ROOT}/{uniq_id}.txt")
        write_txt(f'{ROOT}/{uniq_id}.txt', string_out.strip())
        
        logging.info(f"Writing {ROOT}/{uniq_id}.label")
        write_txt(f'{ROOT}/{uniq_id}.label', '\n'.join(audacity_label_words))


def print_time(string_out, speaker, start_point, end_point, params):
    datetime_offset = 16 * 3600
    if float(start_point) > 3600:
        time_str = "%H:%M:%S.%f"
    else:
        time_str = "%M:%S.%f"
    start_point_str = datetime.fromtimestamp(float(start_point) - datetime_offset).strftime(time_str)[:-4]
    end_point_str = datetime.fromtimestamp(float(end_point) - datetime_offset).strftime(time_str)[:-4]
    strd = "\n[{} - {}] {}: ".format(start_point_str, end_point_str, speaker)
    if params['print_transcript']:
        print(strd, end=" ")
    return string_out + strd


def print_word(string_out, word, params):
    word = word.strip()
    if params['print_transcript']:
        print(word, end=" ")
    return string_out + word + " "


def get_num_of_spk_from_labels(labels):
    spk_set = [x.split(' ')[-1].strip() for x in labels]
    return len(set(spk_set))


def add_json_to_dict(riva_dict, word, stt, end, speaker, params):
    
    riva_dict['words'].append({'word': word,
                                'start_time': stt,
                                'end_time': end,
                                'speaker_label': speaker
                                })
    return riva_dict

def get_speech_labels_from_nonspeech(probs, non_speech, params):
    frame_offset = params['offset'] / params['time_stride']
    speech_labels = []

    if len(non_speech)>0:
        for idx in range(len(non_speech) - 1):
            start = (non_speech[idx][1] + frame_offset) * params['time_stride']
            end = (non_speech[idx + 1][0] + frame_offset) * params['time_stride']
            speech_labels.append("{:.3f} {:.3f} speech".format(start, end))

        if non_speech[-1][1] < len(probs):
            start = (non_speech[-1][1] + frame_offset) * params['time_stride']
            end = (len(probs) + frame_offset) * params['time_stride']
            speech_labels.append("{:.3f} {:.3f} speech".format(start, end))
    else:
        start=0
        end=(len(probs) + frame_offset) * params['time_stride']
        speech_labels.append("{:.3f} {:.3f} speech".format(start, end))


    return speech_labels


def get_file_lists(audiofile_list_path, reference_rttmfile_list_path):
    audio_list, rttm_list = [], []
    with open(audiofile_list_path, 'r') as path2file:
        for audiofile in path2file.readlines():
            audio_list.append(audiofile.strip())
    with open(reference_rttmfile_list_path, 'r') as path2file:
        for rttmfile in path2file.readlines():
            rttm_list.append(rttmfile.strip())
    return audio_list, rttm_list

def write_VAD_rttm_from_speech_labels(ROOT, AUDIO_FILENAME, speech_labels):
    uniq_id = get_uniq_id_from_audio_path(AUDIO_FILENAME)
    with open(f'{ROOT}/{uniq_id}.rttm', 'w') as f:
        for spl in speech_labels:
            start, end, speaker = spl.split()
            start, end = float(start), float(end)
            f.write("SPEAKER {} 1 {:.3f} {:.3f} <NA> <NA> speech <NA>\n".format(uniq_id, start, end - start))

def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum(axis=-1).reshape([logits.shape[0], 1])

def get_audacity_label(word, stt_sec, end_sec, speaker, audacity_label_words):
    spk = speaker.split('_')[-1]
    audacity_label_words.append(f'{stt_sec}\t{end_sec}\t[{spk}] {word}')
    return audacity_label_words

def get_timestamp_in_sec(word_ts_stt_end, params):
    stt = round(params['offset'] + word_ts_stt_end[0] * params['time_stride'], params['round_float'])
    end = round(params['offset'] + word_ts_stt_end[1] * params['time_stride'], params['round_float'])
    return stt, end

def get_transcript_and_logits(audio_file_list):
    with torch.cuda.amp.autocast():
        transcript_logits_list = asr_model.transcribe(
            audio_file_list, batch_size=1, return_text_with_logprobs_and_ts=True
        )
    return transcript_logits_list

def threshold_non_speech(source_list, params):
    non_speech = list(filter(lambda x: x[1] - x[0] > params['threshold'], source_list))
    return non_speech

def get_speech_labels_list(ROOT, transcript_logits_list, audio_file_list, params):
    trans_words_list, spaces_list, word_ts_list = (
        [],
        [],
        [],
    )
    for i, (trans, logit, timestamps) in enumerate(transcript_logits_list):
        AUDIO_FILENAME = audio_file_list[i]
        probs = softmax(logit)
        _spaces, _trans_words = _get_spaces(trans, timestamps)

        blanks = _get_silence_timestamps(probs, symbol_idx=28, state_symbol='blank')
        non_speech = threshold_non_speech(blanks, params)

        speech_labels = get_speech_labels_from_nonspeech(probs, non_speech, params)
        write_VAD_rttm_from_speech_labels(ROOT, AUDIO_FILENAME, speech_labels)
        ipdb.set_trace()

        word_timetamps_middle = [[_spaces[k][1], _spaces[k + 1][0]] for k in range(len(_spaces) - 1)]
        word_timetamps = [[timestamps[0], _spaces[0][0]]] + word_timetamps_middle + [[_spaces[-1][1], logit.shape[0]]]

        word_ts_list.append(word_timetamps)
        spaces_list.append(_spaces)
        trans_words_list.append(_trans_words)

        assert len(_trans_words) == len(word_timetamps)

    return trans_words_list, spaces_list, word_ts_list

def clean_trans_and_TS(trans, timestamps):
    """
    Removes the spaces in the beginning and the end.
    timestamps need to be changed and synced accordingly.
    """
    assert (len(trans) > 0) and (len(timestamps) > 0)
    assert len(trans) == len(timestamps)

    trans = trans.lstrip()
    diff_L= len(timestamps) - len(trans)
    timestamps = timestamps[diff_L:]

    trans = trans.rstrip()
    diff_R = len(timestamps) - len(trans)
    if diff_R > 0:
        timestamps = timestamps[:-1*diff_R]
    return trans, timestamps

def _get_spaces(trans, timestamps):
    trans, timestamps = clean_trans_and_TS(trans, timestamps)
    assert (len(trans) > 0) and (len(timestamps) > 0)
    assert len(trans) == len(timestamps)

    spaces, word_list = [], []
    stt_idx = 0
    for k, s in enumerate(trans):
        if s == ' ':
            spaces.append([timestamps[k], timestamps[k + 1] - 1])
            word_list.append(trans[stt_idx:k])
            stt_idx = k + 1
    if len(trans) > stt_idx and trans[stt_idx] != ' ':
        word_list.append(trans[stt_idx:])
    return spaces, word_list


def write_VAD_rttm(oracle_vad_dir, audio_file_list):
    rttm_file_list = []
    for path_name in audio_file_list:
        uniq_id = get_uniq_id_from_audio_path(path_name)
        rttm_file_list.append(f'{oracle_vad_dir}/{uniq_id}.rttm')

    oracle_manifest = os.path.join(oracle_vad_dir, 'oracle_manifest.json')

    write_rttm2manifest(
        paths2audio_files=audio_file_list, paths2rttm_files=rttm_file_list, manifest_file=oracle_manifest
    )
    return oracle_manifest


def run_diarization(ROOT, audio_file_list, oracle_manifest, oracle_num_speakers, max_num_speakers):

    MODEL_CONFIG = os.path.join(ROOT, 'speaker_diarization.yaml')
    if not os.path.exists(MODEL_CONFIG):
        config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/stable/examples/speaker_recognition/conf/speaker_diarization.yaml"
        MODEL_CONFIG = wget.download(config_url, ROOT)

    config = OmegaConf.load(MODEL_CONFIG)

    pretrained_speaker_model = '/disk2/ejrvs/model_comparision/diarization_ecapa.nemo'
    config.diarizer.paths2audio_files = audio_file_list
    config.diarizer.out_dir = ROOT  # Directory to store intermediate files and prediction outputs
    config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    config.diarizer.speaker_embeddings.oracle_vad_manifest = oracle_manifest
    config.diarizer.oracle_num_speakers = oracle_num_speakers
    config.diarizer.max_num_speakers = max_num_speakers
    config.diarizer.speaker_embeddings.shift_length_in_sec = 0.75
    config.diarizer.speaker_embeddings.window_length_in_sec = 1.5


    oracle_model = ClusteringDiarizer(cfg=config)
    oracle_model.diarize()


def get_uniq_id_from_audio_path(audio_file_path):
    return '.'.join(os.path.basename(audio_file_path).split('.')[:-1])


if __name__ == "__main__":

    audio_file_path = '/data/samsungSSD/NVIDIA/datasets/LibriSpeech_NeMo/LibriSpeech/dev-clean-wav/2803-154328-0013.wav'
    oracle_num_speakers = None
    max_num_speakers = 10
    ROOT = os.path.join(os.getcwd(), 'asr_based_riva_diar')
    os.makedirs(ROOT, exist_ok=True)

    params = {
        "time_stride": 0.02,
        "offset": -0.18,
        "round_float": 3,
        "print_transcript": False,
        "threshold": 30,  # minimun width to consider non-speech activity
    }

    asr_model = EncDecCTCModel4Diar.from_pretrained(model_name='QuartzNet15x5Base-En', strict=False)
    transcript_logits_list = get_transcript_and_logits([audio_file_path])
    word_list, spaces_list, word_ts_list = get_speech_labels_list(
        ROOT, transcript_logits_list, [audio_file_path], params
    )
    oracle_manifest = write_VAD_rttm(ROOT, [audio_file_path])

    run_diarization(ROOT, [audio_file_path], oracle_manifest, oracle_num_speakers, max_num_speakers)

    uniq_id = get_uniq_id_from_audio_path(audio_file_path)
    pred_rttm = os.path.join(ROOT, 'pred_rttms', uniq_id + '.rttm')
    labels = rttm_to_labels(pred_rttm)

    write_json_and_transcript(
        ROOT, [audio_file_path], [labels], word_list, word_ts_list, params
    )
