# Integrate OPT-30B to Chai

# Plan

- [x] Prepare model
- [x] Make a draft of an inference in Colab
- [x] Reduce needed RAM !!! (may be offload state dict)
- [x] Optimize inference with `floatN`
- [x] Test everything with multiple small checkpoints
- [ ] Somehow understand how to convert it to ONNX without 100500 GB of RAM
- [ ] Create some beautiful graphs with benchmarks
- [x] Prepare Dockerfile
- [x] Test locally
- [x] With a help of God upload to CoreWeave
- [ ] !!! SOLVE ISSUE of infinite request !!!


# How to use

1. Change `MODEL_SIZE` in `opt-inference.yaml` and needed memory.
2. ```kubectl apply -f opt-inference.yaml```
3. Inference with this url: http://opt-inference.\<NAMESPACE>.knative.chi.coreweave.com/v1/models/opt:predict