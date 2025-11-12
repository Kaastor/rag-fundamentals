# Security

Never include secrets in prompts. Mask keys using `FROB_***`.
The system stores no user data by default.
To revoke access, rotate your key and update `~/.frob/config.yaml`.
