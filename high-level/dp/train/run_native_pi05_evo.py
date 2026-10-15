"""Scoped launch adapter; native hwk policy/processor/trainer files stay unchanged.

TorchCodec 0.2 (PyTorch 2.6) accepts paths, not hwk's fsspec LocalFileOpener.
Only translate that local file object to its path. Same decoder/seek mode/pixels.
"""
import os
from pathlib import Path
import runpy
import sys


def main():
    import lerobot
    assert '/hwk/lerobot/src/' in lerobot.__file__, lerobot.__file__
    from torchcodec.decoders import VideoDecoder
    original_init = VideoDecoder.__init__

    def compatible_init(self, source, *args, **kwargs):
        if type(source).__module__ == 'fsspec.implementations.local' and hasattr(source, 'path'):
            assert Path(source.path).is_file(), source.path
            source = os.fspath(source.path)
        original_init(self, source, *args, **kwargs)

    VideoDecoder.__init__ = compatible_init
    script = sys.argv.pop(1)
    sys.argv[0] = script
    runpy.run_path(script, run_name='__main__')


if __name__ == '__main__':
    main()
