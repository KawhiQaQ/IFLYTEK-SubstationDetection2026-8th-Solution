# Model weights

The final ensemble requires exactly six checkpoint files. Put them directly in this directory and run:

```bash
python scripts/verify_weights.py --weights weights
```

The exact contract is stored in `manifest.json`. The total is `568,789,683` bytes.

Weights are intentionally excluded from Git. Download the released archive from [Baidu Netdisk](https://pan.baidu.com/s/10u1sjJgp585YQ6dMdfmtYg?pwd=d4hh) with extraction code `d4hh`, then place the six checkpoint files directly in this directory. The manifest is the authoritative identity record.

The competition data are not included. Checkpoint redistribution remains subject to the upstream pretrained-model and competition terms.
