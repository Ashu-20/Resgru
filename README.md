# Resgru Decoder

A recurrent neural network decoder for the **heavy-hexagon quantum error correcting code**, built with [Stim](https://github.com/quantumlib/Stim) and [PyTorch](https://pytorch.org/).

The decoder uses a GRU encoder with a residual MLP head to classify logical errors from syndrome measurement sequences, and supports multi-GPU distributed training via PyTorch DDP.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{Ashu2026resgru,
  author = {Kumar, Ashutosh},
  title  = {Resgru Decoder: A Recurrent Neural Network Decoder for the Heavy-Hexagon Code},
  year   = {2026},
  url    = {https://github.com/Ashu-20/Resgru.git}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
