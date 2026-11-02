# models/ (owners: Uttam, Harshini, Likhitha)

> Placeholder created by the data owner so the folder structure matches the Implementation Architecture Plan. The owner can replace or extend this file.

**Contents:** `credit_card.py` (Uttam), `account_takeover.py` (Harshini), `identity.py` (Likhitha).

Each module provides `build_model(input_dim)` returning a torch `nn.Module`: feedforward network with 2 to 3 hidden
layers, binary cross-entropy with `pos_weight`, roughly 10k to 50k parameters.

Data (see `data/DATA.md`):

```python
from data.loader import load_client_data, load_test, input_dim, get_pos_weight
X_train, y_train, X_val, y_val = load_client_data(bank_id=0, fraud_type="credit_card")
model = build_model(input_dim("credit_card"))   # 95 (credit_card), 10 (account_takeover), 109 (identity_fraud)
pos_w = get_pos_weight(0, "credit_card")
```

Done when: the model trains locally, exposes get and set weights, and baseline F1 and AUC-ROC are logged.
