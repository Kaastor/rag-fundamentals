# Installation

To install Frobnicator CLI, use: `pip install frobnicator`.
Configuration lives in `~/.frob/config.yaml`. Set `api_key` before first use.

After installation, run `frob --help` to view commands.
If the config file does not exist, create it and add:

```yaml
api_key: "<your-api-key>"
mode: "cloud"   # or "offline"
```
