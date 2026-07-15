Merged PR #3101 on `https://github.com/pallets/werkzeug.git`: refactor `EnvironBuilder` file handling and related code

Restore multi-file product behavior against the held-out verifier suite. Affected sources include: src/werkzeug/datastructures/file_storage.py, src/werkzeug/formparser.py, src/werkzeug/test.py, src/werkzeug/wrappers/request.py.
Base commit (immutable): `70551309d170d43696fff527cd5b5893421996ba`.
Do not weaken pass_to_pass coverage. Commit on a branch; leave a clean porcelain tree.
