use clap::Parser;
use grpc_server::cli::Args;
use grpc_server::server::MnistGrpcServer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args = Args::parse();

    // Convert CLI args to server configuration
    let config = args.to_server_config()?;

    // Create and configure the server
    let server = MnistGrpcServer::new(config)?;

    // Initialize tracing
    server.init_tracing()?;

    // Start the server
    server.serve().await?;

    Ok(())
}
