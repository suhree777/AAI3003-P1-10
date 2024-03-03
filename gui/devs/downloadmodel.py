import streamlit as st
from datasets import load_dataset
import pandas as pd
from transformers import pipeline

pipe = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection")