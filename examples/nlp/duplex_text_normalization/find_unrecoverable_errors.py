# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string

try:
    from nemo_text_processing.text_normalization.normalize_with_audio import NormalizerWithAudio

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False

input_file = "/home/ebakhturina/NeMo/examples/nlp/duplex_text_normalization/errors.txt"
input_file = "/home/ebakhturina/misc_scripts/normalization/errors_4000.txt"

normalizer = NormalizerWithAudio(
    input_case='cased', lang='en', cache_dir="/home/ebakhturina/NeMo/examples/nlp/duplex_text_normalization/cache_dir"
)

def remove_punctuation(word: str, remove_spaces=True, do_lower=True):
    """
    Removes all punctuation marks from a word except for '
    that is often a part of word: don't, it's, and so on
    """
    all_punct_marks = string.punctuation
    word = re.sub('[' + all_punct_marks + ']', '', word)

    if remove_spaces:
        word = word.replace(" ", "").replace(u"\u00A0", "").strip()

    if do_lower:
        word = word.lower()
    return word


original_wrong = 0
wrong = 0
correct_with_no_punct = 0
correct_with_no_sil = 0
correct_with_zs = 0
acceptable_error = 0

example = []
with open(input_file.replace(".txt", "_reduced.txt"), 'w') as f_out:
    with open(input_file, 'r') as f:
        for line in f:
            example.append(line)
            if line.startswith('Original Input'):
                _input = line[line.find(':') + 1 :].strip()
            elif line.startswith('Predicted Str'):
                pred = line[line.find(':') + 1 :].strip()
            elif line.startswith('Ground-Truth'):
                target = line[line.find(':') + 1 :].strip()
                original_wrong += 1

                pred_no_punct = remove_punctuation(pred).strip()
                target_no_punct = remove_punctuation(target).strip()
                if pred_no_punct == target_no_punct:
                    correct_with_no_punct += 1
                elif pred_no_punct.replace(" sil ", " ") == target_no_punct:
                    correct_with_no_sil += 1
                elif pred_no_punct.replace("s", "z") == target_no_punct.replace("s", "z"):
                    correct_with_zs += 1
                else:
                    wfst_pred = normalizer.normalize(_input.replace("``", "").strip(), n_tagged=100000)
                    wfst_pred = [remove_punctuation(x) for x in wfst_pred]

                    if pred_no_punct in wfst_pred:
                        acceptable_error += 1
                    else:
                        # print('input: ', _input)
                        # print('nn...:', pred)
                        # print('target:', target)
                        # print("-" * 40)
                        for line in example:
                            f_out.write(line)
                        example = []
                        wrong += 1

print(f'original wrong: {original_wrong}')
print(f'wrong: {wrong}')
print(f'no punct correct: {correct_with_no_punct}')
print(f'no sil correct: {correct_with_no_punct}')
print(f's z correct: {correct_with_zs}')
print(f'Acceptable error: {acceptable_error}')
