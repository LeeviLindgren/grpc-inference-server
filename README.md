# MNIST gRPC Server

This repo contains simple gRPC server implementation for serving MNIST digit classification requests using Rust stack.
Purpose of this project is personal learning and experimentation with [tonic](https://github.com/hyperium/tonic)
for rust implementation of gRPC; and [candle](https://github.com/huggingface/candle), a ml/tensor framework in Rust.

One nice feature of using Rust in ml-inference is the lightweight deployments. While Python deployments with deep learning
frameworks like Pytorch often result in container sizes of multiple GBs, with `candle`,  the release binary size of the
server implementation is only ~6MB (without model weights).

## Model Architectures

This project supports two (rather trivial) neural network architectures for MNIST classification:

1. **Multi-Layer Perceptron (MLP)**
   - 3 fully connected layers: 784 → 128 → 64 → 10
   - Simple feedforward network
   - Good baseline performance

2. **Convolutional Neural Network (ConvNet)**
   - 2 convolutional layers (1→32→64 channels) with ReLU and max pooling
   - 2 fully connected layers: 3136 → 128 → 10
   - Better feature extraction and higher accuracy

These models are defined in the `mnist` sub-crate.

## Usage

### Prerequisites

- Rust (latest stable version)
- Python 3.8+ (for training)
- uv (Python package manager)

### Training the Model

Model training is done in Python with Pytorch:

1. Navigate to the training directory:
   ```bash
   cd training
   ```

2. Install Python dependencies:
   ```bash
   uv sync
   ```

3. Train the model and save weights:
   ```bash
   uv run python train.py --output ../models/mnist_convnet.safetensors
   ```

   The training script will:
   - Download the MNIST dataset automatically
   - Train a ConvNet for 3 epochs
   - Display training progress and final test accuracy
   - Save the model weights in SafeTensors format

### Running the Server

1. Build the server:
   ```bash
   cargo build --release
   ```

2. Start the gRPC server:
   ```bash
   cargo run --release --bin grpc-server -- --model-architecture conv  --model-weights models/mnist_convnet.safetensors
   ```

   The server will start on `[::1]:50051` by default.

You can check other available CLI args with `--help`.

### Getting predictions from the Server

As the protocol expects the images to be sent as raw bytes, one can convert image to base64
and create a request in JSON format:

```bash
echo '{"data": "'$(base64 -w 0 -i ~/Desktop/four.png)'"}' > test_request.json
```

Using [grpcurl](https://github.com/fullstorydev/grpcurl), such requests can be sent to the server:

```bash
grpcurl -plaintext -proto ./proto/mnist.proto \
        -d @ \
        '[::1]:50051' mnist.Mnist.Predict \
        < your_request.json
```

This should respond with something like:

```json
{
  "label": 4,
  "probabilities": [0.001, 0.002, 0.003, 0.004, 0.985, 0.002, 0.001, 0.001, 0.001, 0.000]
}
```
