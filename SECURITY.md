# Reporting a security issue

Use the repository's Security tab to report a vulnerability privately if that
option is available. Otherwise open an issue requesting a private reporting
channel and withhold technical details until a maintainer contacts you.
Include the GameMatch version, installation method, affected component and
observed impact. Use a small generated fixture when sharing a reproduction.
Keep account identifiers, credentials and third-party match records out of
reports.

## Loading models and data

Load checkpoints and joblib preprocessors only from a source you trust.
Serialized Python estimators can execute code when loaded. The completion
inventory detects changes relative to the recorded run. The identity of the
producer must be established separately.

Use `gamematch datasets` to inspect registered sources and `gamematch fetch-data`
to verify their pinned contents. Keep the original provenance when moving a
dataset or distributing a derived artifact. Each dataset retains its own terms.
See the [input contracts](docs/data-contracts.md) and [attribution notices](THIRD_PARTY.md).
