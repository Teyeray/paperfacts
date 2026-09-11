"""Extraction: a parser lane's Markdown becomes sample-level records.

Both lanes run through the same extractor, the same prompt and the same model. The two must stay
byte-for-byte symmetric, otherwise prompt noise leaks into what is supposed to be a measurement of
parser disagreement. The only asymmetry is the Markdown itself.
"""
