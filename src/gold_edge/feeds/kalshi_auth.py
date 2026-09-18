"""Kalshi Trade API v2 request signing.

Signed message = f"{timestamp_ms}{METHOD}{path}" (path includes the
/trade-api/v2 prefix, excludes the query string). Signature is RSA-PSS
over SHA-256, MGF1(SHA-256), salt length == digest length, base64-encoded.
See docs/contract_notes.md for how this was confirmed against live docs.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


class KalshiSigner:
    def __init__(self, api_key_id: str, private_key_path: Path) -> None:
        self.api_key_id = api_key_id
        self._private_key = self._load_private_key(private_key_path)

    @staticmethod
    def _load_private_key(path: Path) -> rsa.RSAPrivateKey:
        key_bytes = path.read_bytes()
        key = serialization.load_pem_private_key(key_bytes, password=None)
        if not isinstance(key, rsa.RSAPrivateKey):
            raise TypeError(f"{path} does not contain an RSA private key")
        return key

    def headers(self, method: str, path: str) -> dict[str, str]:
        """path must include the /trade-api/v2 (or /trade-api/ws/v2) prefix
        and must not include the query string."""
        timestamp_ms = str(int(time.time() * 1000))
        message = f"{timestamp_ms}{method.upper()}{path}".encode()
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=hashes.SHA256().digest_size,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
        }
