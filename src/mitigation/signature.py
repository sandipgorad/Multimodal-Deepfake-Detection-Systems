"""Digital signatures (Ed25519) for content authenticity.

A publisher signs a file's SHA-256 with a private key; anyone with the public
key can confirm the file is unmodified and came from that publisher.
"""
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey)

from src.metadata.metadata_check import sha256_of


def generate_keypair(private_path="keys/private.pem", public_path="keys/public.pem"):
    Path(private_path).parent.mkdir(parents=True, exist_ok=True)
    key = Ed25519PrivateKey.generate()
    Path(private_path).write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()))
    Path(public_path).write_bytes(key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo))


def sign_file(file_path, private_path="keys/private.pem"):
    key = serialization.load_pem_private_key(Path(private_path).read_bytes(), None)
    return key.sign(sha256_of(file_path).encode()).hex()


def verify_file(file_path, signature_hex, public_path="keys/public.pem"):
    pub = serialization.load_pem_public_key(Path(public_path).read_bytes())
    try:
        pub.verify(bytes.fromhex(signature_hex), sha256_of(file_path).encode())
        return True
    except (InvalidSignature, ValueError):
        return False
