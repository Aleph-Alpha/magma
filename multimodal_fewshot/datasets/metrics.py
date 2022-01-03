from typing import Dict, List
from nlgeval import NLGEval

def compute_nlg_metrics(references: List[List[str]], hypothesis: List[str]) -> Dict:

    """
    computes BLEU, METEOR, ROUGE + CIDEr scores.

    Each inner list in references is one set of references for the hypothesis
    (a list of single reference strings for each sentence in hypothesis in the same order).

    Args:
        references: list of lists reference translations.
        hypothesis: list of hypotheses / predictions.

    """

    nlgeval = NLGEval(no_glove=True, no_skipthoughts=True)

    return nlgeval.compute_metrics(references, hypothesis)

