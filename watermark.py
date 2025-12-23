################################################################################################
# Imports
################################################################################################

import os.path
from functools import reduce
from itertools import product
import math
import re
import spacy
import string
import os

import torch

from config import WatermarkArgs, GenericArgs, stop
from models.watermark import InfillModel
from utils.logging import getLogger
from utils.dataset_utils import preprocess2sentence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


################################################################################################
# Helper functions for watermarking process
################################################################################################

def evaluate_watermark_likelihood(candidate_message: str, true_message: str) -> float:
    # Pad candidate and true messsages so that they are the same length
    if len(candidate_message) > len(true_message):
        no_padding_bits = len(candidate_message) - len(true_message)
        true_message = "0" * no_padding_bits + true_message

    elif len(candidate_message) < len(true_message):
        no_padding_bits = len(true_message) - len(candidate_message)
        candidate_message = "0" * no_padding_bits + candidate_message

    # Compare messaged bit-wise
    total_bits = len(true_message)
    matching_bits = 0

    for candidate_bit, true_bit in zip(list(candidate_message), list(true_message)):
        if candidate_bit == true_bit:
            matching_bits += 1

    # Convert matching bits into a percentage
    watermark_likelihood = (matching_bits / total_bits) * 100

    return watermark_likelihood


################################################################################################
# Watermark a text with a binary message
################################################################################################

def embed(input_text, message, generic_args, infill_args):
    start_sample_idx = 0
    DEBUG_MODE = generic_args.debug_mode
    dtype = generic_args.dtype

    input_texts = preprocess2sentence(
        [input_text],
        corpus_name=dtype,
        start_sample_idx=start_sample_idx,
        cutoff_q=(0.0, 1.0),
        use_cache=False
    )


    # You can add your own entity / keyword that should NOT be masked.
    # This list will need to be saved when extracting
    # infill_args.custom_keywords = ["watermarking", "watermark"]

    spacy_tokenizer = spacy.load(generic_args.spacy_model)
    if "trf" in generic_args.spacy_model:
        spacy.require_gpu()

    model = InfillModel(infill_args)

    bit_count = 0
    word_count = 0

    logger = getLogger(
        "DEMO",
        debug_mode=DEBUG_MODE
    )

    logger.info(f"Starting watermarking process...")

    for formatted_text in input_texts:
        print(f"Formatted texts: {formatted_text}")
        corpus_level_watermarks = []

        for s_idx, sentence in enumerate(formatted_text):
            sentence = spacy_tokenizer(sentence.text.strip())
            all_keywords, entity_keywords = model.keyword_module.extract_keyword([sentence])
            keyword = all_keywords[0]
            ent_keyword = entity_keywords[0]

            agg_cwi, agg_probs, tokenized_pt, (mask_idx_pt, mask_idx, mask_word) = model.run_iter(sentence, keyword, ent_keyword,
                                                                                                train_flag=False, embed_flag=True)
            # check if keyword & mask_indices matches
            valid_watermarks = []
            valid_watermarks_print = []
            tokenized_text = [token.text_with_ws for token in sentence]

            if len(agg_cwi) > 0:
                for cwi in product(*agg_cwi):
                    wm_text = tokenized_text.copy()
                    for m_idx, c_id in zip(mask_idx, cwi):
                        wm_text[m_idx] = re.sub(r"\S+", model.tokenizer.decode(c_id), wm_text[m_idx])

                    wm_tokenized = spacy_tokenizer("".join(wm_text).strip())

                    # extract keyword of watermark
                    wm_keywords, wm_ent_keywords = model.keyword_module.extract_keyword([wm_tokenized])
                    wm_kwd = wm_keywords[0]
                    wm_ent_kwd = wm_ent_keywords[0]
                    wm_mask_idx, wm_mask = model.mask_selector.return_mask(wm_tokenized, wm_kwd, wm_ent_kwd)  # type: ignore

                    # checking whether the watermark can be embedded without the assumption of corruption
                    mask_match_flag = len(wm_mask) and set(wm_mask_idx) == set(mask_idx)
                    if mask_match_flag:
                        text2print = [t.text_with_ws for t in wm_tokenized]
                        for m_idx in mask_idx:
                            text2print[m_idx] = f"\033[92m{text2print[m_idx]}\033[00m"

                        valid_watermarks_print.append(text2print)
                        valid_watermarks.append(wm_tokenized.text)

            punct_removed = sentence.text.translate(str.maketrans(dict.fromkeys(string.punctuation)))
            word_count += len([i for i in punct_removed.split(" ") if i not in stop])

            # Log watermarked sentence to console
            logger.info(f"***Sentence {s_idx}***")
            if len(valid_watermarks_print) > 1:
                bit_count += math.log2(len(valid_watermarks))

                for vw in valid_watermarks_print:
                    logger.info("".join(vw))

            if len(valid_watermarks) == 0:
                valid_watermarks = [sentence.text]

            corpus_level_watermarks.append(valid_watermarks)

        if word_count:
            logger.info(f"Average message bits per word: {bit_count / word_count:.3f}")

        num_options = reduce(lambda x, y: x * y, [len(vw) for vw in corpus_level_watermarks])
        available_bit = math.floor(math.log2(num_options))
        logger.info(f"No. watermark options: {num_options}")
        logger.info(f"No. bits that can be embedded: {available_bit}")

        # left pad to available bit if given message is short
        if available_bit > 8:
            logger.info(f"Available bit is large: {available_bit} > 8.. "
                f"We recommend using shorter text segments as it may take a while")

        message = "0" * (available_bit - len(message)) + message
        if len(message) > available_bit:
            logger.info(f"Given message longer than capacity. Truncating...: {len(message)}>{available_bit}")
            message = message[:available_bit]

        message_decimal = int(message, 2)
        cnt = 0
        available_candidates = product(*corpus_level_watermarks)
        watermarked_sentences = next(available_candidates)

        while cnt < message_decimal:
            cnt += 1
            watermarked_sentences = next(available_candidates)

        logger.info("---- Watermarked text ----")
        full_watermarked_text = ""
        for wt in watermarked_sentences:
            sentence = "".join(wt)
            full_watermarked_text += " " + sentence
            logger.info(sentence)

        full_watermarked_text = full_watermarked_text.strip()
        logger.info(f"Full watermarked text: {full_watermarked_text}")

        return full_watermarked_text


################################################################################################
# Detect if a text is watermarked
################################################################################################

def extract(input_text, generic_args, infill_args):
    DEBUG_MODE = generic_args.debug_mode

    dtype = generic_args.dtype

    watermarked_text = preprocess2sentence(
        [input_text],
        corpus_name=dtype,
        start_sample_idx=0,
        cutoff_q=(0.0, 1.0),
        use_cache=False
    )
    watermarked_text = watermarked_text[0]

    spacy_tokenizer = spacy.load(generic_args.spacy_model)
    if "trf" in generic_args.spacy_model:
        spacy.require_gpu()

    model = InfillModel(infill_args)

    logger = getLogger(
        "EXTRACT",
        debug_mode=DEBUG_MODE
    )

    decoded_message = []

    for wm_sentence in watermarked_text:
        wm_sentence = str(wm_sentence)
        logger.info(f"Sentence to be checked: {wm_sentence}")

        sentence = spacy_tokenizer(wm_sentence)
        all_keywords, entity_keywords = model.keyword_module.extract_keyword([sentence])

        # Extracting states for (potentially corrupted) watermarked text
        keyword = all_keywords[0]
        ent_keyword = entity_keywords[0]
        agg_cwi, agg_probs, tokenized_pt, (mask_idx_pt, mask_idx, mask_word) = model.run_iter(sentence, keyword, ent_keyword,
                                                                                            train_flag=False, embed_flag=True)

        wm_keys = model.tokenizer(" ".join([t.text for t in mask_word]), add_special_tokens=False)['input_ids']

        valid_watermarks = []
        valid_keys = []
        tokenized_text = [token.text_with_ws for token in sentence]

        if len(agg_cwi) > 0:
            for cwi in product(*agg_cwi):
                wm_text = tokenized_text.copy()
                for m_idx, c_id in zip(mask_idx, cwi):
                    wm_text[m_idx] = re.sub(r"\S+", model.tokenizer.decode(c_id), wm_text[m_idx])

                wm_tokenized = spacy_tokenizer("".join(wm_text).strip())

                # Extract keyword of watermark
                wm_keywords, wm_ent_keywords = model.keyword_module.extract_keyword([wm_tokenized])
                wm_kwd = wm_keywords[0]
                wm_ent_kwd = wm_ent_keywords[0]
                wm_mask_idx, wm_mask = model.mask_selector.return_mask(wm_tokenized, wm_kwd, wm_ent_kwd)  # type: ignore

                # Checking whether the watermark can be embedded
                mask_match_flag = len(wm_mask) > 0 and set(wm_mask_idx) == set(mask_idx)
                if mask_match_flag:
                    valid_watermarks.append(wm_tokenized.text)
                    valid_keys.append(torch.stack(cwi).tolist())

        extracted_msg = []
        if len(valid_keys) > 1:
            try:
                extracted_msg_decimal = valid_keys.index(wm_keys)
            except:
                similarity = [len(set(wm_keys).intersection(x)) for x in valid_keys]
                similar_key = max(zip(valid_keys, similarity), key=lambda x: x[1])[0]
                extracted_msg_decimal = valid_keys.index(similar_key)

            num_digit = math.ceil(math.log2(len(valid_keys)))
            extracted_msg = format(extracted_msg_decimal, f"0{num_digit}b")
            extracted_msg = list(map(int, extracted_msg))

        logger.info(F"Extracted message bits: {extracted_msg}")
        decoded_message.extend(extracted_msg)

    binary_message = "".join([str(bit) for bit in decoded_message])
    logger.info(f"Full extracted message: {binary_message}")

    return binary_message


if __name__ == '__main__':
    # Parse input arguments
    generic_parser = GenericArgs()
    generic_args, _ = generic_parser.parse_known_args()

    infill_parser = WatermarkArgs()
    infill_args, _ = infill_parser.parse_known_args()

    infill_args.dtype = "custom"
    infill_args.exp_name = "tmp"

    infill_args.mask_select_method = "grammar"
    infill_args.mask_order_by = "dep"

    infill_args.model_name = "bert-large-cased"
    infill_args.exclude_cc = True
    infill_args.topk = 2
    infill_args.keyword_mask = "na"

    # Run watermarking
    print("\n============================ WATERMARK EMBEDDING PROCESS ============================\n")

    raw_text = "Recent years have witnessed a proliferation of valuable original natural language contents found in subscription-based media outlets, web novel platforms, and outputs of large language models. However, these contents are susceptible to illegal piracy and potential misuse without proper security measures. This calls for a secure watermarking system to guarantee copyright protection through leakage tracing or ownership identification. To effectively combat piracy and protect copyrights, a multi-bit watermarking framework should be able to embed adequate bits of information and extract the watermarks in a robust manner despite possible corruption. In this work, we explore ways to advance both payload and robustness by following a well-known proposition from image watermarking and identify features in natural language that are invariant to minor corruption. Through a systematic analysis of the possible sources of errors, we further propose a corruption-resistant infill model. Our full method improves upon the previous work on robustness by +16.8% point on average on four datasets, three corruption types, and two corruption ratios."  # noqa
    message = "1010101"

    watermarked_text = embed(raw_text, message, generic_args, infill_args)

    # Run extraction
    print("\n\n============================ WATERMARK EXTRACTION PROCESS ============================\n")

    extracted_message = extract(watermarked_text, generic_args, infill_args)
    watermarked_likelihood = evaluate_watermark_likelihood(extracted_message, message)
    print(f"\nLikelihood of text being watermarked: {watermarked_likelihood}%")
