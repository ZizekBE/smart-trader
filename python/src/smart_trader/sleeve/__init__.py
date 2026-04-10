"""Hybrid strategy sleeve package.

Two sleeves run in parallel on the same portfolio:
  - core     (long-term): high capital, low frequency, wide trailing stop
  - tactical (short-term): low capital per trade, high frequency, fixed TP + time stop
"""
