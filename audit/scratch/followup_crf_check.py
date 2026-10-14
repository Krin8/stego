"""Import/dispatch check only; no inference, installation or result files."""
import fixes2_guard as g
import sys,os,tempfile,importlib
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(g.ROOT/'src'))
os.environ['MPLCONFIGDIR']=str(g.SCRATCH/'mpl');tempfile.tempdir=str(g.SCRATCH/'tmp')
try:
    module=importlib.import_module('eval_segmentation')
except ModuleNotFoundError as e:
    assert e.name.startswith('pydensecrf') and '--expect-fixed' not in sys.argv
    print('BEFORE: non-CRF evaluation import fails:',e)
else:
    assert 'crf' not in sys.modules and not any(n.startswith('pydensecrf') for n in sys.modules)
    with patch.object(module.os,'makedirs') as mkdir,patch.object(module,'_load_crf',side_effect=AssertionError('CRF imported while disabled')):
        module.my_app.__wrapped__(SimpleNamespace(run_crf=False,pytorch_data_dir='unused',experiment_name='check',model_paths=[]))
    try:module.my_app.__wrapped__(SimpleNamespace(run_crf=True))
    except RuntimeError as e:assert 'run_crf=True requires pydensecrf' in str(e)
    else:raise AssertionError('Missing CRF dependency did not fail clearly')
    print('AFTER PASS: import and run_crf=False need no CRF; run_crf=True fails clearly before outputs/model loading')
