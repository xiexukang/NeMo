#!/usr/bin/env python
# coding: utf-8

import csv
import copy
import json
import os
from collections import OrderedDict as od
from datetime import datetime
from math import ceil
from typing import Dict, List, Optional, Union
from functools import reduce
import argparse
import ipdb
import librosa
import matplotlib.pyplot as plt

import numpy as np
import torch
import wget
from omegaconf import OmegaConf
from torchmetrics import Metric
from tqdm.auto import tqdm
from pyannote.metrics.diarization import DiarizationErrorRate

from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.models import ClusteringDiarizer, EncDecCTCModel

from nemo.collections.asr.parts.utils.speaker_utils import audio_rttm_map as get_audio_rttm_map
from nemo.collections.asr.parts.utils.speaker_utils import (
    get_DER,
    labels_to_pyannote_object,
    rttm_to_labels,
    write_rttm2manifest,
)
from nemo.utils import logging

class WER_TS(WER):
    def __init__(
        self,
        vocabulary,
        batch_dim_index=0,
        use_cer=False,
        ctc_decode=True,
        log_prediction=True,
        dist_sync_on_step=False,
    ):
        super().__init__(
        vocabulary,
        batch_dim_index,
        use_cer,
        ctc_decode,
        log_prediction,
        dist_sync_on_step)

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
        return token_list, timestamp_list

    def ctc_decoder_predictions_tensor_with_ts(
        self,
        predictions: torch.Tensor,
        predictions_len: torch.Tensor = None,
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

            text, timestamp_list = self.decode_tokens_to_str_with_ts(decoded_prediction, decoded_timing_list)
            hypotheses.append(text)
            timestamps.append(timestamp_list)

        return hypotheses, timestamps


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
        spaces.append([idx_state, len(probs)-1])

    return spaces


def dump_json_to_file(file_path, riva_dict):
    with open(file_path, "w") as outfile:
        json.dump(riva_dict, outfile, indent=4)


def write_txt(w_path, val):
    with open(w_path, "w") as output:
        output.write(val + '\n')
    return None

def get_DER(all_reference, all_hypothesis):
    metric = DiarizationErrorRate(collar=0.5, skip_overlap=True, uem=None)
    
    mapping_dict = {}
    for k, (reference, hypothesis) in enumerate(zip(all_reference, all_hypothesis)):
        metric(reference, hypothesis, detailed=True)
        mapping_dict[k] = metric.optimal_mapping(reference, hypothesis)

    DER = abs(metric)
    CER = metric['confusion'] / metric['total']
    FA = metric['false alarm'] / metric['total']
    MISS = metric['missed detection'] / metric['total']
    
    metric.reset()

    return DER, CER, FA, MISS, mapping_dict

def write_json_and_transcript(
    ROOT,
    audio_file_list,
    transcript_logits_list,
    diar_labels,
    word_list,
    word_ts_list,
    spaces_list,
    params,
):
    total_riva_dict = {}
    for k, audio_file_path in enumerate(audio_file_list):
        uniq_id = get_uniq_id_from_audio_path(audio_file_path)
        labels, spaces = diar_labels[k], spaces_list[k]
        audacity_label_words = []
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
            space_stt_end = [word_ts_stt_end[1], word_ts_stt_end[1]] if j == len(spaces) else spaces[j]
            trans, logits, timestamps = transcript_logits_list[k]

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
            )  
            
            total_riva_dict[uniq_id] = riva_dict
            audacity_label_words = get_audacity_label(words[j], 
                                                      stt_sec, end_sec,
                                                      speaker,
                                                      audacity_label_words)
        
        write_and_log(ROOT, uniq_id, riva_dict, string_out, audacity_label_words)

    return total_riva_dict

def write_and_log(ROOT, uniq_id, riva_dict, string_out, audacity_label_words):
    logging.info(f"Writing {ROOT}/json_result/{uniq_id}.json")
    dump_json_to_file(f'{ROOT}/json_result/{uniq_id}.json', riva_dict)
    
    logging.info(f"Writing {ROOT}/trans_with_spks{uniq_id}.txt")
    write_txt(f'{ROOT}/trans_with_spks/{uniq_id}.txt', string_out.strip())
    
    logging.info(f"Writing {ROOT}/audacity_label/{uniq_id}.w.label")
    write_txt(f'{ROOT}/audacity_label/{uniq_id}.w.label', '\n'.join(audacity_label_words))


def isOverlapArray(rangeA, rangeB):
    startA, endA = rangeA[:, 0], rangeA[:, 1]
    startB, endB = rangeB[:, 0], rangeB[:, 1]
    return (endA > startB) & (endB > startA)

def getOverlapRangeArray(rangeA, rangeB):
    left = np.max(np.vstack((rangeA[:, 0], rangeB[:, 0])), axis=0)
    right = np.min(np.vstack((rangeA[:, 1], rangeB[:, 1])), axis=0)
    return right-left

def get_timestamp_in_sec(word_ts_stt_end, params):
    stt = round(params['offset'] + word_ts_stt_end[0] * params['time_stride'], params['round_float'])
    end = round(params['offset'] + word_ts_stt_end[1] * params['time_stride'], params['round_float'])
    return stt, end

def get_audacity_label(word, stt_sec, end_sec, speaker, audacity_label_words):
    spk = speaker.split('_')[-1]
    audacity_label_words.append(f'{stt_sec}\t{end_sec}\t[{spk}] {word}')
    return audacity_label_words

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

def write_VAD_rttm_from_speech_labels(ROOT, AUDIO_FILENAME, speech_labels, params):
    uniq_id = get_uniq_id_from_audio_path(AUDIO_FILENAME)
    with open(f'{ROOT}/oracle_vad/{uniq_id}.rttm', 'w') as f:
        for spl in speech_labels:
            start, end, speaker = spl.split()
            start, end = float(start), float(end)
            f.write("SPEAKER {} 1 {:.3f} {:.3f} <NA> <NA> speech <NA>\n".format(uniq_id, start, end - start))

def get_file_lists(audiofile_list_path, reference_rttmfile_list_path):
    audio_list, rttm_list = [], []
    
    if not audiofile_list_path or (audiofile_list_path in ['None', 'none', 'null', '']):
        raise ValueError("audiofile_list_path is not provided.")
    else:
        with open(audiofile_list_path, 'r') as path2file:
            for audiofile in path2file.readlines():
                uniq_id = get_uniq_id_from_audio_path(audiofile)
                # if uniq_id in SELECTED:
                audio_list.append(audiofile.strip())
  
    if reference_rttmfile_list_path != None and (not (reference_rttmfile_list_path in ['None', 'none', 'null', ''])):
        with open(reference_rttmfile_list_path, 'r') as path2file:
            for rttmfile in path2file.readlines():
                # uniq_id = get_uniq_id_from_audio_path(rttmfile)
                # if uniq_id in SELECTED:
                rttm_list.append(rttmfile.strip())

    return audio_list, rttm_list


def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum(axis=-1).reshape([logits.shape[0], 1])


def get_transcript_and_logits(audio_file_list, _asr_model):
    trans_logit_timestamps_list = []
    wer_ts = WER_TS(
        vocabulary=_asr_model.decoder.vocabulary,
        batch_dim_index=0,
        use_cer=_asr_model._cfg.get('use_cer', False),
        ctc_decode=True,
        dist_sync_on_step=True,
        log_prediction=_asr_model._cfg.get("log_prediction", False),
        )
    # audio_file_list, batch_size=1, return_text_with_logprobs_and_ts=True
    with torch.cuda.amp.autocast():
        transcript_logits_list = _asr_model.transcribe(
            audio_file_list, batch_size=1, logprobs=True
        )
        for logit_np in transcript_logits_list:
            log_prob = torch.from_numpy(logit_np)
            logits_len = torch.from_numpy(np.array([log_prob.shape[0]]))
            greedy_predictions = log_prob.argmax(dim=-1, keepdim=False).unsqueeze(0)
            text, ts = wer_ts.ctc_decoder_predictions_tensor_with_ts(greedy_predictions, 
                                                                     predictions_len=logits_len)
            trans_logit_timestamps_list.append([text[0], logit_np, ts[0]])
    return trans_logit_timestamps_list

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

        # Only for quartznet! Citrinet's "space" symbol is different and not reliable
        _spaces, _trans_words = _get_spaces(trans, timestamps)
        
        if not params['external_oracle_vad']:
            blanks = _get_silence_timestamps(probs, symbol_idx=28, state_symbol='blank')
            non_speech = threshold_non_speech(blanks, params)
        
            speech_labels = get_speech_labels_from_nonspeech(probs, non_speech, params)
            write_VAD_rttm_from_speech_labels(ROOT, AUDIO_FILENAME, speech_labels, params)
        
        word_timetamps_middle = [[_spaces[k][1], _spaces[k + 1][0]] for k in range(len(_spaces) - 1)]
        word_timetamps = [[timestamps[0], _spaces[0][0]]] + word_timetamps_middle + [[_spaces[-1][1], logit.shape[0]]]

        word_ts_list.append(word_timetamps)
        spaces_list.append(_spaces)
        trans_words_list.append(_trans_words)

        assert len(_trans_words) == len(word_timetamps)

    return trans_words_list, spaces_list, word_ts_list

def clean_trans_and_ts(trans, timestamps):
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
    trans, timestamps = clean_trans_and_ts(trans, timestamps)
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


def run_diarization(ROOT, audio_file_list, oracle_manifest, oracle_num_speakers, pretrained_speaker_model):
    if oracle_num_speakers != None:
        if oracle_num_speakers.isnumeric():
            oracle_num_speakers = int(oracle_num_speakers)
        elif oracle_num_speakers in ['None', 'none', 'Null', 'null', '']:
            oracle_num_speakers = None


    data_dir = os.path.join(ROOT, 'data')

    MODEL_CONFIG = os.path.join(data_dir, 'speaker_diarization.yaml')
    if not os.path.exists(MODEL_CONFIG):
        config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/stable/examples/speaker_recognition/conf/speaker_diarization.yaml"
        MODEL_CONFIG = wget.download(config_url, data_dir)

    config = OmegaConf.load(MODEL_CONFIG)

    output_dir = os.path.join(ROOT, 'oracle_vad')
    config.diarizer.paths2audio_files = audio_file_list
    config.diarizer.out_dir = output_dir  # Directory to store intermediate files and prediction outputs
    config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    config.diarizer.speaker_embeddings.oracle_vad_manifest = oracle_manifest
    config.diarizer.oracle_num_speakers = oracle_num_speakers
    config.diarizer.speaker_embeddings.shift_length_in_sec = 0.75
    config.diarizer.speaker_embeddings.window_length_in_sec = 1.0
    oracle_model = ClusteringDiarizer(cfg=config)
    oracle_model.diarize()


def get_uniq_id_from_audio_path(audio_file_path):
    return '.'.join(os.path.basename(audio_file_path).split('.')[:-1])

def eval_diarization(audio_file_list, rttm_file_list, output_dir):
    diar_labels = []
    DER_result_dict = {}
    all_hypotheses = []
    all_references = []
    ref_labels_list = []
    count_correct_spk_counting = 0

    if rttm_file_list == []: 
        for k, audio_file_path in enumerate(audio_file_list):
            uniq_id = get_uniq_id_from_audio_path(audio_file_path)
            pred_rttm = os.path.join(output_dir, 'pred_rttms', uniq_id + '.rttm')
            pred_labels = rttm_to_labels(pred_rttm)
            diar_labels.append(pred_labels)
            est_n_spk = get_num_of_spk_from_labels(pred_labels)
            logging.info(f"Estimated n_spk [{uniq_id}]: {est_n_spk}")

        return diar_labels, None, None

    else: 
        audio_rttm_map = get_audio_rttm_map(audio_file_list, rttm_file_list)
        for k, audio_file_path in enumerate(audio_file_list):
            uniq_id = get_uniq_id_from_audio_path(audio_file_path)
            rttm_file = audio_rttm_map[uniq_id]['rttm_path']
            if os.path.exists(rttm_file):
                ref_labels = rttm_to_labels(rttm_file)
                ref_labels_list.append(ref_labels)
                reference = labels_to_pyannote_object(ref_labels)
                all_references.append(reference)
            else:
                raise ValueError("No reference RTTM file provided.")

            pred_rttm = os.path.join(output_dir, 'pred_rttms', uniq_id + '.rttm')
            pred_labels = rttm_to_labels(pred_rttm)
            diar_labels.append(pred_labels)

            est_n_spk = get_num_of_spk_from_labels(pred_labels)
            ref_n_spk = get_num_of_spk_from_labels(ref_labels)
            hypothesis = labels_to_pyannote_object(pred_labels)
            all_hypotheses.append(hypothesis)
            DER, CER, FA, MISS, mapping = get_DER([reference], [hypothesis])
            DER_result_dict[uniq_id] = {"DER": DER, "CER": CER, "FA": FA, "MISS": MISS, "n_spk": est_n_spk, "mapping": mapping[0], "spk_counting": (est_n_spk == ref_n_spk) }
            count_correct_spk_counting += int(est_n_spk == ref_n_spk)

        DER, CER, FA, MISS, mapping = get_DER(all_references, all_hypotheses)
        logging.info(
            "Cumulative results of all the files:  \n FA: {:.4f}\t MISS {:.4f}\t\
                Diarization ER: {:.4f}\t, Confusion ER:{:.4f}".format(
                FA, MISS, DER, CER
            )
        )
        DER_result_dict['total'] = {"DER": DER, "CER": CER, "FA": FA, "MISS": MISS, "spk_counting_acc": count_correct_spk_counting/len(audio_file_list)}
        return diar_labels, ref_labels_list, DER_result_dict

def get_WDER(total_riva_dict, DER_result_dict, audio_file_list, ref_labels_list, params):
    
    wder_dict = {}
    grand_total_word_count = 0
    grand_correct_word_count = 0
    for k, audio_file_path in enumerate(audio_file_list):
        
        labels = ref_labels_list[k]
        uniq_id = get_uniq_id_from_audio_path(audio_file_path)
        try:
            mapping_dict = DER_result_dict[uniq_id]['mapping']
        except:
            ipdb.set_trace()
        words_list = total_riva_dict[uniq_id]['words']
        
        idx = 0
        total_word_count = len(words_list) 
        correct_word_count = 0
        ref_label_list = [ [float(x.split()[0]), float(x.split()[1])] for x in labels ] 
        ref_label_array = np.array(ref_label_list)

        for wdict in words_list:
            speaker_label = wdict['speaker_label']
            if speaker_label in mapping_dict:
                est_spk_label = mapping_dict[speaker_label]
            else:
                continue
            start_point, end_point, ref_spk_label = labels[idx].split()
            word_range = np.array([wdict['start_time'], wdict['end_time']])
            word_range_tile = np.tile(word_range, (ref_label_array.shape[0], 1))
            ovl_bool = isOverlapArray(ref_label_array, word_range_tile)
            if np.any(ovl_bool) == False:
                continue

            ovl_length = getOverlapRangeArray(ref_label_array, word_range_tile)
            
            if params['lenient_overlap_WDER']:
                ovl_length_list = list(ovl_length[ovl_bool])
                max_ovl_sub_idx = np.where(ovl_length_list == np.max(ovl_length_list))[0]
                max_ovl_idx = np.where(ovl_bool==True)[0][max_ovl_sub_idx]
                ref_spk_labels = [ x.split()[-1] for x in list(np.array(labels)[max_ovl_idx]) ]
                if est_spk_label in ref_spk_labels:
                    correct_word_count += 1
            else: 
                max_ovl_sub_idx = np.argmax(ovl_length[ovl_bool])
                max_ovl_idx = np.where(ovl_bool==True)[0][max_ovl_sub_idx]
                _, _, ref_spk_label = labels[max_ovl_idx].split()
                correct_word_count += int(est_spk_label == ref_spk_label)

        wder= 1 - (correct_word_count/total_word_count)
        grand_total_word_count += total_word_count
        grand_correct_word_count += correct_word_count

        wder_dict[uniq_id] = wder

    wder_dict['total'] = 1 - (grand_correct_word_count/grand_total_word_count)
    print("Total WDER: ", wder_dict['total'])

    return wder_dict
        
def write_result_in_csv(args, WDER_dict, DER_result_dict, effective_WDER):
    
    row = [
    args.threshold, 
    WDER_dict['total'], 
    DER_result_dict['total']['DER'],
    DER_result_dict['total']['FA'],
    DER_result_dict['total']['MISS'],
    DER_result_dict['total']['CER'],
    DER_result_dict['total']['spk_counting_acc'],
    effective_WDER
    ]

    with open(os.path.join(ROOT,args.csv), 'a') as csvfile: 
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(row) 
                

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_speaker_model", type=str, help="Fullpath of the Speaker embedding extractor model (*.nemo).", required=True)
    parser.add_argument("--audiofile_list_path", type=str, help="Fullpath of a file contains the list of audio files", required=True)
    parser.add_argument("--reference_rttmfile_list_path", type=str, help="Fullpath of a file contains the list of rttm files")
    parser.add_argument("--oracle_vad_manifest", type=str, help="External VAD file for diarization")
    parser.add_argument("--oracle_num_speakers", help="Either int or text file that contains number of speakers")
    parser.add_argument("--threshold", default=50, type=int, help="Threshold for ASR based VAD")
    parser.add_argument("--csv", default='result.csv', type=str, help="")

    args = parser.parse_args()

    """
    CH109: All sessions have two speakers.

    python get_json_ASR_and_diarization.py \
    --audiofile_list_path='/disk2/scps/audio_scps/callhome_ch109.scp' \
    --reference_rttmfile_list_path='/disk2/scps/rttm_scps/callhome_ch109.rttm' \
    --oracle_num_speakers=2

    AMI: Oracle number of speakers in EN2002c.Mix-Lapel is 3, not 4.

    python get_json_ASR_and_diarization.py \
    --audiofile_list_path='/disk2/datasets/amicorpus/mixheadset_test_wav.list' \
    --reference_rttmfile_list_path='/disk2/datasets/amicorpus/mixheadset_test_rttm.list' \
    --oracle_num_speakers=2

    """
    
    ROOT = os.path.join(os.getcwd(), 'asr_based_diar')
    oracle_vad_dir = os.path.join(ROOT, 'oracle_vad')
    json_result = (os.path.join(ROOT, 'json_result'))
    trans_with_spks = os.path.join(ROOT, 'trans_with_spks')
    audacity_label = os.path.join(ROOT, 'audacity_label')
    
    os.makedirs(ROOT, exist_ok=True)
    os.makedirs(oracle_vad_dir, exist_ok=True)
    os.makedirs(json_result, exist_ok=True)
    os.makedirs(trans_with_spks, exist_ok=True)
    os.makedirs(audacity_label, exist_ok=True)

    data_dir = os.path.join(ROOT, 'data')
    os.makedirs(data_dir, exist_ok=True)

    params = {
        "time_stride": 0.02, # This should not be changed if you are using QuartzNet15x5Base.
        "offset": -0.18, # This should not be changed if you are using QuartzNet15x5Base.
        "round_float": 2,
        "print_transcript": False,
        "lenient_overlap_WDER": True, #False,
        "threshold": args.threshold,  # minimun width to consider non-speech activity
        "external_oracle_vad": True if args.oracle_vad_manifest else False,
    }

    asr_model = EncDecCTCModel.from_pretrained(model_name='QuartzNet15x5Base-En', strict=False)

    audio_file_list, rttm_file_list = get_file_lists(args.audiofile_list_path, args.reference_rttmfile_list_path)

    transcript_logits_list = get_transcript_and_logits(audio_file_list, asr_model)

    word_list, spaces_list, word_ts_list = get_speech_labels_list(
        ROOT, transcript_logits_list, audio_file_list, params
    )

    if not args.oracle_vad_manifest:
        oracle_manifest = write_VAD_rttm(oracle_vad_dir, audio_file_list)
    else:
        oracle_manifest = args.oracle_vad_manifest

    run_diarization(ROOT, audio_file_list, oracle_manifest, args.oracle_num_speakers, args.pretrained_speaker_model)

    diar_labels, ref_labels_list, DER_result_dict = eval_diarization(audio_file_list, rttm_file_list, oracle_vad_dir)

    total_riva_dict = write_json_and_transcript(
                                            ROOT,
                                            audio_file_list,
                                            transcript_logits_list,
                                            diar_labels,
                                            word_list,
                                            word_ts_list,
                                            spaces_list,
                                            params,
                                        )
    if rttm_file_list != []:
        
        WDER_dict = get_WDER(total_riva_dict, DER_result_dict, audio_file_list, ref_labels_list, params)
        
        effective_WDER = 1 - ((1 - (DER_result_dict['total']['FA'] + DER_result_dict['total']['MISS'])) * (1 - WDER_dict['total']))

        logging.info(f" total \nWDER : {WDER_dict['total']:.4f} \
                              \nDER  : {DER_result_dict['total']['DER']:.4f} \
                              \nFA   : {DER_result_dict['total']['FA']:.4f} \
                              \nMISS : {DER_result_dict['total']['MISS']:.4f} \
                              \nCER  : {DER_result_dict['total']['CER']:.4f} \
                              \nspk_counting_acc : {DER_result_dict['total']['spk_counting_acc']:.4f} \
                              \neffective_WDER : {effective_WDER:.4f}")

        write_result_in_csv(args, WDER_dict, DER_result_dict, effective_WDER)
