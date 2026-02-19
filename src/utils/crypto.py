import base64
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


def get_or_create_key() -> bytes:
    key_file = "data/.secret_key"
    os.makedirs(os.path.dirname(key_file), exist_ok=True)
    
    if os.path.exists(key_file):
        with open(key_file, 'rb') as f:
            return f.read()
    
    key = Fernet.generate_key()
    with open(key_file, 'wb') as f:
        f.write(key)
    
    return key


class Crypto:
    def __init__(self):
        self.key = get_or_create_key()
        self.fernet = Fernet(self.key)
    
    def encrypt(self, data: str) -> str:
        if not data:
            return ""
        encrypted = self.fernet.encrypt(data.encode('utf-8'))
        return base64.b64encode(encrypted).decode('utf-8')
    
    def decrypt(self, encrypted_data: str) -> str:
        if not encrypted_data:
            return ""
        try:
            decoded = base64.b64decode(encrypted_data.encode('utf-8'))
            decrypted = self.fernet.decrypt(decoded)
            return decrypted.decode('utf-8')
        except Exception:
            return ""


_crypto_instance = None


def get_crypto() -> Crypto:
    global _crypto_instance
    if _crypto_instance is None:
        _crypto_instance = Crypto()
    return _crypto_instance


def encrypt(data: str) -> str:
    return get_crypto().encrypt(data)


def decrypt(encrypted_data: str) -> str:
    return get_crypto().decrypt(encrypted_data)
