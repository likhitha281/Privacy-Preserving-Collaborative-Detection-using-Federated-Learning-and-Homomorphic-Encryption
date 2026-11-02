# he/ (owner: Anjan)

> Placeholder created by the data owner so the folder structure matches the Implementation Architecture Plan. The owner can replace or extend this file.

**Contents:** `ckks.py`, `keys.py`, `dp.py`: TenSEAL CKKS context and key setup, chunked encrypt, aggregate and decrypt,
optional update clipping and DP noise, cost measurements.

Done when: the decrypted aggregate matches the plaintext average within a small tolerance, and time and ciphertext
size per round are reported.
