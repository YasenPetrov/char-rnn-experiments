import sys
import os

# TODO: Fix sibling directory imports
cwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cwd, '../..'))

from lmexperiments.datautils.dataset import TextFile, Alphabet, _TOKEN_UNK, _TOKEN_PAD, _ID_UNK, _ID_PAD

# TODO: Test predicting text probability should be independent of batch size

# TODO: Test save/load model

# TODO: Test can reproduce training results