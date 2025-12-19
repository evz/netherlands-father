#!/bin/bash
set -e

# Open-Sora Server Setup Script
# Run this on the remote machine with the GPU

INSTALL_DIR="${OPENSORA_DIR:-/opt/Open-Sora}"
SERVER_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Open-Sora Server Setup ==="
echo "Install directory: $INSTALL_DIR"
echo "Server directory: $SERVER_DIR"
echo ""

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Please install NVIDIA drivers."
    exit 1
fi

echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Clone Open-Sora if not exists
if [ ! -d "$INSTALL_DIR" ]; then
    echo "=== Cloning Open-Sora ==="
    sudo git clone https://github.com/hpcaitech/Open-Sora "$INSTALL_DIR"
    sudo chown -R $USER:$USER "$INSTALL_DIR"
else
    echo "=== Open-Sora already cloned, pulling latest ==="
    cd "$INSTALL_DIR"
    git pull
fi

# Create virtual environment
echo "=== Setting up Python environment ==="
cd "$INSTALL_DIR"
if [ ! -d "venv" ]; then
    python3.10 -m venv venv
fi
source venv/bin/activate

# Install Open-Sora
echo "=== Installing Open-Sora ==="
pip install --upgrade pip
pip install -v .

# Install CUDA dependencies (adjust for your CUDA version)
echo "=== Installing CUDA dependencies ==="
pip install xformers --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn --no-build-isolation || echo "flash-attn install failed, continuing..."

# Install server dependencies
echo "=== Installing server dependencies ==="
pip install -r "$SERVER_DIR/requirements.txt"

# Download model checkpoints
echo "=== Downloading model checkpoints ==="
pip install "huggingface_hub[cli]"
huggingface-cli download hpcai-tech/Open-Sora-v2 --local-dir "$INSTALL_DIR/ckpts"

# Create systemd service
echo "=== Creating systemd service ==="
sudo tee /etc/systemd/system/opensora-server.service > /dev/null <<EOF
[Unit]
Description=Open-Sora Video Generation Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$SERVER_DIR
Environment="OPENSORA_DIR=$INSTALL_DIR"
Environment="PATH=$INSTALL_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$INSTALL_DIR/venv/bin/python opensora_server.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable opensora-server

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start the server:"
echo "  sudo systemctl start opensora-server"
echo ""
echo "To check status:"
echo "  sudo systemctl status opensora-server"
echo ""
echo "To view logs:"
echo "  journalctl -u opensora-server -f"
echo ""
echo "Server will be available at http://0.0.0.0:8000"
echo ""
echo "To start manually for testing:"
echo "  cd $SERVER_DIR"
echo "  source $INSTALL_DIR/venv/bin/activate"
echo "  OPENSORA_DIR=$INSTALL_DIR python opensora_server.py"
