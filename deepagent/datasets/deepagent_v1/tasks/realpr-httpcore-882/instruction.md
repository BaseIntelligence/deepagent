Merged PR #882 on `https://github.com/encode/httpcore.git`: Fix support for connection Upgrade and CONNECT when some data in the stream has been read.

Restore multi-file product behavior against the held-out verifier suite. Affected sources include: httpcore/_async/http11.py, httpcore/_sync/http11.py.
Base commit (immutable): `c46802478cdd8a82ee8cb333420080fab1aed00b`.
Do not weaken pass_to_pass coverage. Commit on a branch; leave a clean porcelain tree.
