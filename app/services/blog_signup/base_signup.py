# app/services/blog_signup/base_signup.py
import random
import string

def generate_random_email(domain="mailinator.com") -> str:
    local = ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))
    return f"{local}@{domain}"

def generate_username(prefix="user") -> str:
    return f"{prefix}_{random.randint(1000, 9999)}"

def generate_password(length=12) -> str:
    chars = string.ascii_letters + string.digits + "!@#$"
    return ''.join(random.choices(chars, k=length))
