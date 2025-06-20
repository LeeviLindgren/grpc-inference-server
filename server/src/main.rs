use server::app::{AppConfig, WeightsProvider, app};

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().init();

    let config = AppConfig {
        device: candle_core::Device::Cpu,
        dtype: candle_core::DType::F32,
        weights_provider: WeightsProvider::LocalFile("models/mnist_mlp.safetensors".to_string()),
    };
    let app = app(config);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:8080")
        .await
        .unwrap();
    axum::serve(listener, app).await.unwrap();
}
