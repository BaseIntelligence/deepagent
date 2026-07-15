Merged PR #4070 on `https://github.com/Textualize/rich.git`: perf: reduce Console and RichHandler import time by deferring unused imports

Restore multi-file product behavior against the held-out verifier suite. Affected sources include: rich/_emoji_replace.py, rich/console.py, rich/emoji.py, rich/logging.py, rich/protocol.py, rich/repr.py, rich/segment.py, rich/syntax.py.
Base commit (immutable): `fc41075a3206d2a5fd846c6f41c4d2becab814fa`.
Do not weaken pass_to_pass coverage. Commit on a branch; leave a clean porcelain tree.
