"""Locate non-code resources consumed by Melo's English pronunciation path."""
from pathlib import Path
import importlib.util


def english_text_resources(melo_code_path):
    import nltk
    resources = {}
    def required(name, path):
        path = Path(path).expanduser().resolve()
        if not path.exists() or (path.is_file() and not path.stat().st_size):
            raise FileNotFoundError(f'OpenVoice English text resource is missing or empty: {path}')
        resources[name] = path
    text = Path(melo_code_path) / 'melo' / 'text'
    required('melo_cmudict', text / 'cmudict.rep')
    if (text / 'cmudict_cache.pickle').exists():
        required('melo_cmudict_cache', text / 'cmudict_cache.pickle')
    spec = importlib.util.find_spec('g2p_en')
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError('OpenVoice English text processing requires g2p_en')
    required('g2p_checkpoint', Path(spec.origin).parent / 'checkpoint20.npz')
    # g2p_en checks these archives at import and may attempt a download if absent.
    for name, resource in (('nltk_cmudict_archive', 'corpora/cmudict.zip'),
                           ('nltk_tagger_archive', 'taggers/averaged_perceptron_tagger.zip')):
        pointer = nltk.data.find(resource)
        archive = getattr(getattr(pointer, 'zipfile', None), 'filename', None)
        required(name, archive or str(pointer))
    # Resolve the corpus actually used, which can differ from the archive above.
    nltk.corpus.cmudict.ensure_loaded()
    pointer = nltk.corpus.cmudict.root
    archive = getattr(getattr(pointer, 'zipfile', None), 'filename', None)
    required('nltk_cmudict_active', archive or str(pointer))
    # Modern NLTK uses a separate English tagger; older versions use the legacy
    # directory. Probe the same tagger factory as pos_tag without downloading.
    tagger = nltk.tag._get_tagger()
    modern = hasattr(tagger, 'load_from_json')
    pointer = nltk.data.find('taggers/averaged_perceptron_tagger_eng/' if modern
                             else 'taggers/averaged_perceptron_tagger/')
    archive = getattr(getattr(pointer, 'zipfile', None), 'filename', None)
    required('nltk_tagger_active', archive or str(pointer))
    return resources
