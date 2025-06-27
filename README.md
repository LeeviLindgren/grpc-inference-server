

## Rest server

```sh
curl -H "Content-Type: application/json" \
        -d @examples/data/request.example.json \
        http://localhost:8080/predict
```

## Grpc server

Without gRPC reflection:

```sh
grpcurl -plaintext -proto ./proto/mnist.proto \
        -d @ \
        '[::1]:50051' mnist.Mnist.Predict \
        < grpc-server/examples/data/request.json
```

Proto compatible json to send with grpcurl:

```sh
echo '{"data": "'$(base64 -w 0 -i ~/Desktop/four.png)'"}' > data/requests/four.json
```
