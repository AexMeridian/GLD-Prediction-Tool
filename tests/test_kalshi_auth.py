from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from gold_edge.feeds.kalshi_auth import KalshiSigner


def _write_test_key(tmp_path: Path) -> tuple[Path, rsa.RSAPrivateKey]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    key_path = tmp_path / "test_key.pem"
    key_path.write_bytes(pem)
    return key_path, key


def test_headers_contain_expected_keys(tmp_path):
    key_path, _ = _write_test_key(tmp_path)
    signer = KalshiSigner("test-key-id", key_path)
    headers = signer.headers("GET", "/trade-api/v2/markets")

    assert headers["KALSHI-ACCESS-KEY"] == "test-key-id"
    assert headers["KALSHI-ACCESS-TIMESTAMP"].isdigit()
    assert len(headers["KALSHI-ACCESS-SIGNATURE"]) > 0


def test_signature_verifies_against_public_key(tmp_path):
    import base64

    key_path, private_key = _write_test_key(tmp_path)
    signer = KalshiSigner("test-key-id", key_path)
    headers = signer.headers("GET", "/trade-api/v2/portfolio/balance")

    message = (
        headers["KALSHI-ACCESS-TIMESTAMP"] + "GET" + "/trade-api/v2/portfolio/balance"
    ).encode()
    signature = base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"])
    public_key = private_key.public_key()

    # Raises InvalidSignature if verification fails.
    public_key.verify(
        signature,
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=hashes.SHA256().digest_size,
        ),
        hashes.SHA256(),
    )


def test_signature_changes_with_path(tmp_path):
    key_path, _ = _write_test_key(tmp_path)
    signer = KalshiSigner("test-key-id", key_path)
    h1 = signer.headers("GET", "/trade-api/v2/markets")
    h2 = signer.headers("GET", "/trade-api/v2/portfolio/balance")
    assert h1["KALSHI-ACCESS-SIGNATURE"] != h2["KALSHI-ACCESS-SIGNATURE"]


def test_rejects_non_rsa_key(tmp_path):
    from cryptography.hazmat.primitives.asymmetric import ed25519

    key = ed25519.Ed25519PrivateKey.generate()
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    key_path = tmp_path / "ed25519_key.pem"
    key_path.write_bytes(pem)

    try:
        KalshiSigner("test-key-id", key_path)
        raise AssertionError("expected TypeError for non-RSA key")
    except TypeError:
        pass
