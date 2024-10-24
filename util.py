import shelve
import signal
from contextlib import contextmanager
from functools import wraps
from inspect import signature
from typing import Callable, Optional, Tuple, List


def memoize_instance_method(method):
    cache = {}

    def wrapper(self, *args, **kwargs):
        key = (args, tuple(kwargs.items()))
        if key not in cache:
            cache[key] = method(self, *args, **kwargs)
        return cache[key]

    return wrapper


def use_defaults_on_none(func):
    sig = signature(func)
    defaults = {
        k: v.default for k, v in sig.parameters.items() if v.default is not v.empty
    }
    print(defaults)

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Bind the provided arguments to the function signature
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Use the default value for any argument that is None
        for arg_name, arg_value in bound_args.arguments.items():
            if arg_value is None and arg_name in defaults:
                bound_args.arguments[arg_name] = defaults[arg_name]

        return func(*bound_args.args, **bound_args.kwargs)

    return wrapper


def parse_bool_string(string_value: str) -> bool:
    parsed_value = string_value.strip().lower()
    if parsed_value == "true":
        return True
    elif parsed_value == "false":
        return False
    else:
        raise ValueError(f"Invalid boolean string: {string_value}")


def count_lines_starting_with_zero_dot(input_string):
    lines = input_string.split("\n")
    count = 0
    for line in lines:
        if line.strip().startswith("0."):
            count += 1
    return count


class AutoComputedShelfDB:
    def __init__(
        self,
        filename: str,
        compute_default_value: Callable[[str], str],
        flags="c",
        mode=0o666,
    ):
        self.db: shelve.Shelf[str] = shelve.open(filename)
        self._encoding = "utf-8"
        self._compute_default_value = compute_default_value

    @staticmethod
    def _encode(value: Optional[str]) -> str:
        return value.strip()

    @staticmethod
    def _decode(value: str) -> str:
        return value

    def __getitem__(self, key: str) -> str:
        encoded_key = self._encode(key)
        try:
            value = self._decode(self.db.__getitem__(encoded_key))
        except KeyError:
            value = self._compute_default_value(encoded_key)
            self.db.__setitem__(encoded_key, self._encode(value))
            self.db.sync()
        return value

    def __setitem__(self, key: str, value: str):
        self.db.__setitem__(self._encode(key), self._encode(value))
        self.db.sync()

    def __delitem__(self, key: str):
        self.db.__delitem__(self._encode(key))

    def get(self, key: str, default: str = None) -> str:
        encoded_key: str = self._encode(key)
        try:
            value: str = self._decode(self.db.__getitem__(encoded_key))
        except KeyError:
            if default is not None:
                value = default
            else:
                value = self._compute_default_value(encoded_key)
            self.db.__setitem__(encoded_key, self._encode(value))
            self.db.sync()
        return value

    def pop(self, key: str, default=None) -> str:
        return self._decode(super().pop(self._encode(key), self._encode(default)))

    def popitem(self) -> Tuple[str, str]:
        key: bytes
        value: bytes
        key, value = self.db.popitem()
        return self._decode(key), self._decode(value)


def join_strings_with_dots(strings: List[str]):
    result = ""

    counter: int = 0

    for i, string in enumerate(strings):
        if string.lstrip(" .") == "":
            continue
        if counter == 0:
            result += string
        elif result.endswith("."):
            result += " " + string
        else:
            result += ". " + string
        counter += 1

    return result


def find_nth_occurrence(substring: str, string: str, n: int):
    """
    Finds the nth occurrence of 'substring' in 'string'.
    Returns the index of the nth occurrence, or -1 if not found.
    """
    index = -1
    for _ in range(n):
        # Find the next occurrence of the substring
        index = string.find(substring, index + 1)
        # If the substring is not found, return -1
        if index == -1:
            return -1
    return index


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def tokenized_length(text, tokenizer):
    return len(tokenizer.encode(text, return_tensors="pt", truncation=False).squeeze())


def join_strings_with_period(strings: List[str]):
    if not strings:
        return ""
    result = strings[0]
    for s in strings[1:]:
        if not result.endswith("."):
            result += "."
        result += " " + s.rstrip(" . \n")
    return result

hf_local_files_only: bool = False
task_data_v2_str: str = """
1\tFalse\t1\t\tCell line name\tName of the cell line used **for this particular sample**, preferably as a single word.  Barb does not include particular modifications introduced in this study. She outputs "primary" if tissue was used instead of an established [immortalized] cell line, or "N/A" if no reference to a specific cell line is provided
2\tFalse\t2\t1\tCell type\tCell type (e.g. fibroblast, cardiomyoblast, monocyte, adenocarcinoma, etc.), as noted in the record or as inferred by Barb from the cell line name. Barb checks for any typos (e.g. "epitheilal" instead of "epithelial") and corrects them
3\tFalse\t3\t1,2\tOrgan\tOrgan of origin denoted in the record or inferred by Barb from the cell line name, preferably as a single word, using the most common term (e.g., lung, PBMC, liver, cornea, ovary, breast)
4\tFalse\t4\t1,2,3\tIntra-organ location\tMore detailed location within the organ (e.g. right atrium auricular region, bronchus, etc.)
5\tFalse\t5\t\tGenetic modifications\tGenetic modifications (e.g. gene knockout, shRNA or RNAi knockdown or silencing, etc.) introduced by the experimenters **for this particular sample**, with names of genes targeted (if any), and excluding wild-type ("WT") genes
6\tFalse\t6\t\tInput control\tDoes the string "input" appear anywhere in the sample name? Is the sample an input control?
7\tFalse\t7\t1,5\tCell name or abbreviation appears in sample name\tDoes the full name of the cells used, or an abbreviation of that name, appear in the sample name?

8\tTrue\t-1\t\tAntibody catalog numbers and manufacturer strings\tQuote any catalog numbers, lot numbers, and manufacturers exactly as they appear in the record (e.g. "Santa Cruz, C-20, sc-1008, lot# H1216")
9\tFalse\t-1\t\tAntibody catalog references\tAntibody catalog references in record, formatted as e.g. manufacturer=santa_cruz,clone=C-20,catalog=sc-1008,lot=H1216,target=VDR
10\tFalse\t-1\t\tHuman gene names or protein complexes mentioned in record\tQuote any human gene names, or human protein complexes, exactly as they appear in the record. If the same gene is mentioned in different ways, choose the form corresponding to the standardized symbol (e.g., prefer "AR" over "Androgen receptor", or "ESR1" over "ER-alpha").

11\tFalse\t8\t1,5,6,7\tBarb's rationale for ChIP target extraction\tBarb's rationale for ChIP target extraction **for this particular sample** from the record and from Barb's own understanding, or for identification as an "input" / empty-vector (not expressing tagged protein) sample. Barb includes the strategy for protein tagging, if relevant, but ignores genetic modifications (e.g. Cas9 gene editing) or genetic background or genetic modifications that do not involve protein tagging of ChIP targets. She thinks step by step, pays particular attention to the sample name, and repeats record entries providing the information as well as words present in the sample name that refer to the ChIP target or "input" and not to the genetic background.
12\tFalse\t9\t6,7,11\tChIP target\tName of ChIP target **for this particular sample**, or "input" if this is an "input" control sample (as indicated, e.g., by the sample name), or if the targeted tag was not actually expressed (e.g., empty vector)
13\tFalse\t10\t12\tHGNC official gene name for ChIP target\tHGNC official human gene name for ChIP target, or "Unsure" if the official name does not appear consistent with the context of the experiment
14\tFalse\t11\t11,12\tSample is generic ChIP-seq\tDoes this sample correspond to generic ChIP-seq? (Barb answers as: ChIP-seq for sure / No, it may be [ATAC-seq, RNA-seq, etc.] / Unsure.)
15\tFalse\t12\t5,11,12\tBarb's rationale for notable treatment extraction\tBarb's rationale for identification of notable treatments applied **to this particular sample** OTHER THAN any genetic modifications (knockout, knockdown, silencing, etc.) already reported above by Barb and OTHER THAN those related to crosslinking, library preparation and sequencing, regular cell culture, etc. Barb includes references to the record entries providing the information, and to relevant words present in the sample name, including possibly "control" if that refers to a *treatment* control instead of a *ChIP input* control; if applicable, Barb compares the sample name to the names of the other samples in the study to identify abbreviations showing which samples had the treatment applied and which did not
16\tFalse\t13\t5,15\tNotable treatments\tNotable treatments applied to **this particular sample** OTHER THAN genetic modifications (knockout, knockdown, silencing, etc.) already reported above, and OTHER THAN those related to crosslinking, library preparation and sequencing, regular cell culture, etc., and formatted as e.g. "cisplatin (concentration=2_uM, duration=3_days, details=DNA_alkylating_agent)". Barb does not report treatments that don't seem to make sense.
17\tFalse\t14\t5,6,15,16\tThis sample received a control genetic modification or has a control genetic background\tDoes this sample correspond to a control genetic modification, or control genetic background? If so, Barb also names the genetic background/modification to which it should be compared.
18\tFalse\t15\t5,6,15,16,17\tThis sample received a control treatment\tDoes this sample correspond to a control **treatment** (other than genetic modification or background), for comparison with a different treatment in the same experiment? If so, Barb also names that different treatment.
19\tFalse\t16\t\tLow-level gene ontology terms\tLow-level gene ontology terms for biological processes Barb can infer for this experiment. Barb does not report generic processes such as histone or chromatin modification, or "Gene Regulation", "Gene expression", "Transcription", "Chromatin Accessibility", "Epigenetic regulation", "Remodeling", etc. and focuses instead on more specific processes such as "DNA damage repair", "Response to hypoxia", "Response to viral infection", "Brain development", etc
20\tFalse\t17\t19\tRelationship to COVID/pneumonia/inflammation/DNA damage\tIs this sample related to COVID/pneumonia/inflammation/DNA damage? (Barb answers as: Yes / No / Unsure)
"""
barb_header: str = """
Barb is a biologist analyzing metadata from a ChIP-seq experiment database. Her task is to extract information from\
 a record describing a single sample that is part of a larger study. The record may contain incomplete or misorganized\
 metadata, and it's Barb's job to identify the protein that was targeted in the ChIP experiment and to extract\
 information about the sample.

The record is:
```
"""
barb_footer: str = """```

Barb parses all of the information above to complete the following (she outputs "N/A" or "Unsure" where appropriate). \
Unless a concise answer is requested, she thinks step by step and details her reasoning. Barb provides concise, \
professional, insightful, helpful, and truthful explanations for her answers.

"""
BARB_QA_EOL_MARKER: str = ""
PROTOCOL_PARAGRAPH_HEADER: str = "The protocol information in this paragraph likely"
LAST_SENTENCE_MARKER: str = "We used siRNA to knock down Notch1"
answer_perplexities: shelve.Shelf[List[float]] = shelve.open(
    "answer_perplexities", writeback=True
)
IGNORE_INDEX: int = -100
DEFAULT_PAD_TOKEN: str = "<pad>"
DEFAULT_EOS_TOKEN: str = "</s>"
DEFAULT_BOS_TOKEN: str = "<s>"
DEFAULT_UNK_TOKEN: str = "<unk>"
TENTATIVE_MAX_LENGTH_BEFORE_COMPLETION: int = 2048 - 130
shortened_bob_prompt_base: str = """
Bob is an expert biologist analyzing sentences from a database record describing a ChIP-seq experiment. Bob needs to identify sentences that contain information about ChIP targets, cells processed, or treatments applied to those cells. This will help downstream text analysis to be performed in the future. Bob is not interested in fine technical detail, as his purpose is not to reproduce the experiments or to optimize them. Bob is also not **at all** interested in the technical aspect of the ChIP protocol. To perform his task, Bob outputs a numbered list of Yes/No answers about each sentence:
1. Is this sentence of interest to Bob?
2. Does it correspond to scientific background of the study, or to interpretation of its results?
3. Does it contain a file name with substrings (possibly abbreviated) that refer to sample-specific antibodies or their targets, cell line names, drugs, or treatment conditions?
4. Does it pertain solely to metadata?
5. Does it mention the specific antibodies used for IP, their catalogue numbers or manufacturers, or how they were raised?
6. Does it add **new** information (not already included in preceding sentences) about the cell line, tissue, or organ used for ChIP, or about the gene expression, overexpression or silencing status, or vectors the cells may contain?
7. Does it mention "interesting" cell treatments including e.g. drug treatments, application of stress or stimuli, or drugs to induce expression? Bob is not interested in regular cell culture techniques or cell preparation for ChIP.

Bob provides concise, professional, insightful, helpful, and truthful explanations for his answers.

Bob now analyzes *one by one* all the sentences in the text below.
```
"""
shortened_bob_prompt_base_with_example: str = """
Bob is an expert biologist analyzing sentences from a database record describing a ChIP-seq experiment. Bob's needs to identify sentences that contain information about ChIP targets, cells processed, or treatments applied to those cells. This will help downstream text analysis to be performed in the future. Bob is not interested in fine technical detail, as his purpose is not to reproduce the experiments or to optimize them. Bob is also not **at all** interested in the technical aspect of the ChIP protocol. To perform his task, Bob outputs a numbered list of Yes/No answers about each sentence:
1. Is this sentence of interest to Bob?
2. Does it correspond to scientific background of the study, or to interpretation of its results?
3. Does it contain a file name with substrings (possibly abbreviated) that refer to sample-specific antibodies or their targets, cell line names, drugs, or treatment conditions?
4. Does it pertain solely to metadata?
5. Does it mention the specific antibodies used for IP, their catalogue numbers or manufacturers, or how they were raised?
6. Does it add **new** information (not already included in preceding sentences) about the cell line, tissue, or organ used for ChIP, or about the gene expression, overexpression or silencing status, or vectors the cells may contain?
7. Does it mention "interesting" cell treatments including e.g. drug treatments, application of stress or stimuli, or drugs to induce expression? Bob is not interested in regular cell culture techniques or cell preparation for ChIP.

Bob provides concise, professional, insightful, helpful, and truthful explanations for his answers, as shown in the following example:

Sentence:
The second day, after 2 washes with RIPA-0.5, 1 wash with RIPA-0.3, 1 wash with RIPA-0, 2 washes with LiCl buffer (10 mM Tris-HCl, 0.25 M LiCl, 0.25% NP-40, and 0,25% NaDOC, pH7.4), and 2 washes with TE buffer, bound protein-DNA complexes were resuspended in elution buffer (10 mM Tris-HCl, 1mM EDTA, and 1% SDS, pH7.4) supplemented with 10 µg/ml RNase A for elution and RNA digestion, and incubated at 55 °C for 1 hour.
Bob's explanation:
The sentence describes protocol details of no relevance (hence 1:No) and gives no information about antibodies (hence 5:No), or cell genetic background (hence 6:No), cell treatments (hence 7:No), etc.
Bob's answer:
1:No  2:No  3:No  4:No  5:No  6:No  7:No  ###END

Bob now analyzes *one by one* all the sentences in the text below.
```
"""
model_name = "huggyllama/llama-30b"
model_size = '30B'
num_added_tokens: int = 0


