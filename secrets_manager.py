# secret_manager.py
import streamlit as st

def get_secret(key, default=None):
    return st.secrets.get(key, default)
