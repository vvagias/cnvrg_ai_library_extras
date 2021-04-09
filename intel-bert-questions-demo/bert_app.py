#!/usr/bin/env python3

"""
 Copyright (c) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import logging as log
import sys
import time
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
import os
import numpy as np
from openvino.inference_engine import IECore

#sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
from tokens_bert import text_to_tokens, load_vocab_file
from html_reader import get_paragraphs
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-v", "--vocab", help="Required. path to the vocabulary file with tokens",
                      required=False, default="bert-small-uncased-whole-word-masking-squad-int8-0002/vocab.txt", type=str)
    args.add_argument("-m", "--model", default=Path("bert-large-uncased-whole-word-masking-squad-0001/FP16/bert-large-uncased-whole-word-masking-squad-0001.xml"),  help="Required. Path to an .xml file with a trained model",
                      required=False, type=Path)
    args.add_argument("-i", "--input",  help="Required. URL to a page with context",
                      action='append', required=False, type=str, default=["https://en.wikipedia.org/wiki/Bert_(Sesame_Street)"])
    args.add_argument("--questions", type=str, nargs='+', metavar='QUESTION', help="Optional. Prepared questions")
    args.add_argument("--input_names",
                      help="Optional. Inputs names for the network. "
                           "Default values are \"input_ids,attention_mask,token_type_ids\" ",
                      required=False, type=str, default="0, 1, 2")
    args.add_argument("--output_names",
                      help="Optional. Outputs names for the network. "
                           "Default values are \"output_s,output_e\" ",
                      required=False, type=str, default="3171, 3172")
    args.add_argument("--model_squad_ver", help="Optional. SQUAD version used for model fine tuning",
                      default="1.2", required=False, type=str)
    args.add_argument("-q", "--max_question_token_num", help="Optional. Maximum number of tokens in question",
                      default=8, required=False, type=int)
    args.add_argument("-a", "--max_answer_token_num", help="Optional. Maximum number of tokens in answer",
                      default=15, required=False, type=int)
    args.add_argument("-d", "--device",
                      help="Optional. Target device to perform inference on."
                           "Default value is CPU",
                      default="CPU", type=str)
    args.add_argument('-r', '--reshape', action='store_true',
                      help="Optional. Auto reshape sequence length to the "
                           "input context + max question len (to improve the speed)")
    args.add_argument('-c', '--colors', action='store_true',
                      help="Optional. Nice coloring of the questions/answers. "
                           "Might not work on some terminals (like Windows* cmd console)")
    return parser


# return entire sentence as start-end positions for a given answer (within the sentence).
def find_sentence_range(context, s, e):
    # find start of sentence
    for c_s in range(s, max(-1, s - 200), -1):
        if context[c_s] in "\n.":
            c_s += 1
            break

    # find end of sentence
    for c_e in range(max(0, e - 1), min(len(context), e + 200), +1):
        if context[c_e] in "\n.":
            break

    return c_s, c_e

def setup(url):
    global vocab
    global ie_encoder
    global input_names
    global output_names
    global model
    global c_tokens_id
    global ie_encoder_exec
    global args
    global c_tokens_se
    global context
    global COLOR_RED
    global COLOR_RESET



    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    if args.colors:
        COLOR_RED = "\033[91m"
        COLOR_RESET = "\033[0m"
    else:
        COLOR_RED = ""
        COLOR_RESET = ""

    # load vocabulary file for model
    log.info("Loading vocab file:\t{}".format(args.vocab))

    vocab = load_vocab_file(args.vocab)
    log.info("{} tokens loaded".format(len(vocab)))

    # get context as a string (as we might need it's length for the sequence reshape)
    p = url
    paragraphs = get_paragraphs([p])
    context = '\n'.join(paragraphs)
    log.info("Size: {} chars".format(len(context)))
    log.info("Context: " + COLOR_RED + context + COLOR_RESET)
    # encode context into token ids list
    c_tokens_id, c_tokens_se = text_to_tokens(context.lower(), vocab)

    log.info("Initializing Inference Engine")
    ie = IECore()
    version = ie.get_versions(args.device)[args.device]
    version_str = "{}.{}.{}".format(version.major, version.minor, version.build_number)
    log.info("Plugin version is {}".format(version_str))

    # read IR
    model_xml = args.model
    model_bin = model_xml.with_suffix(".bin")
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))

    ie_encoder = ie.read_network(model=model_xml, weights=model_bin)

    if args.reshape:
        # reshape the sequence length to the context + maximum question length (in tokens)
        first_input_layer = next(iter(ie_encoder.inputs))
        c = ie_encoder.inputs[first_input_layer].shape[1]
        # find the closest multiple of 64, if it is smaller than current network's sequence length, let' use that
        seq = min(c, int(np.ceil((len(c_tokens_id) + args.max_question_token_num) / 64) * 64))
        if seq < c:
            input_info = list(ie_encoder.inputs)
            new_shapes = dict([])
            for i in input_info:
                n, c = ie_encoder.inputs[i].shape
                new_shapes[i] = [n, seq]
                log.info("Reshaped input {} from {} to the {}".format(i, ie_encoder.inputs[i].shape, new_shapes[i]))
            log.info("Attempting to reshape the network to the modified inputs...")
            try:
                ie_encoder.reshape(new_shapes)
                log.info("Successful!")
            except RuntimeError:
                log.error("Failed to reshape the network, please retry the demo without '-r' option")
                sys.exit(-1)
        else:
            log.info("Skipping network reshaping,"
                     " as (context length + max question length) exceeds the current (input) network sequence length")

    # check input and output names
    input_names = list(i.strip() for i in args.input_names.split(','))
    output_names = list(o.strip() for o in args.output_names.split(','))
    if ie_encoder.inputs.keys() != set(input_names) or ie_encoder.outputs.keys() != set(output_names):
        log.error("Input or Output names do not match")
        log.error("    The demo expects input->output names: {}->{}. "
                  "Please use the --input_names and --output_names to specify the right names "
                  "(see actual values below)".format(input_names, output_names))
        log.error("    Actual network input->output names: {}->{}".format(list(ie_encoder.inputs.keys()),
                                                                          list(ie_encoder.outputs.keys())))
        log.error("    Actual network input->output values: {}->{}".format(list(ie_encoder.inputs.values()),
                                                                          list(ie_encoder.outputs.values())))
        raise Exception("Unexpected network input or output names")


    # load model to the device
    log.info("Loading model to the {}".format(args.device))
    ie_encoder_exec = ie.load_network(network=ie_encoder, device_name=args.device)
