# Troubleshooting

If you see `AUTH_MISSING`, create the config and set `api_key`.
If you see `RATE_LIMIT`, retry with exponential backoff.
For performance issues lower `level` or use batch mode.
If outputs drift in style, set `level` back to the default (3).
