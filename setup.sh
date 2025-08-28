#!/bin/bash

# Setup script for development environment
set -e

echo "🚀 Setting up development environment..."

# Check and set locale if needed
echo "📍 Checking locale settings..."
if ! locale | grep -q "LANG=en_US.UTF-8"; then
    echo "Setting up en_US.UTF-8 locale..."
    sudo sed -i 's/^# *en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen
    sudo locale-gen
    sudo update-locale LANG=en_US.UTF-8
    echo "✅ Locale set to en_US.UTF-8"
else
    echo "✅ Locale already set to en_US.UTF-8"
fi

# Install packages
echo "📦 Installing system packages..."
sudo apt update
sudo apt install -y neovim ranger gh tree btop

# Install Python packages
echo "🐍 Installing Python packages..."
pip install nvitop

# Install Node.js packages
echo "📦 Installing Node.js packages..."
npm install -g @anthropic-ai/claude-code

# Create tmux.conf
echo "⚙️ Creating ~/.tmux.conf..."
cat > ~/.tmux.conf << 'EOF'
set -g default-terminal "tmux-256color"
set -as terminal-features ",*:RGB"
set -s escape-time 0
unbind C-b
set -g prefix C-a
bind C-a send-prefix
EOF

# Create inputrc
echo "⚙️ Creating ~/.inputrc..."
cat > ~/.inputrc << 'EOF'
set keyseq-timeout 0
EOF

# Configure git
git config --global credential.helper store
echo "❗ Please set git user and email:"
echo "   git config --global user.name 'Your Name'"
echo "   git config --global user.email 'your.email@example.com'"

# Add vi mode to bashrc if not already present
echo "⚙️ Checking vi mode in bashrc..."
if ! grep -q "set -o vi" ~/.bashrc; then
    echo "set -o vi" >> ~/.bashrc
    echo "✅ Added vi mode to bashrc"
else
    echo "✅ Vi mode already enabled in bashrc"
fi

echo "🎉 Setup complete! Please run 'source ~/.bashrc' or restart your shell."
echo ""
echo "Installed:"
echo "  - neovim, ranger, gh, tree, btop"
echo "  - nvitop (Python package)"
echo "  - @anthropic-ai/claude-code (Node.js package)"
echo "  - tmux configuration"
echo "  - inputrc configuration"
echo "  - vi mode for bash"
echo "  - en_US.UTF-8 locale"
echo ""
echo "❗ REMINDERS:"
echo "  1. Set up git user and email (see above)"
echo "  2. Log in to Hugging Face: huggingface-cli login"
echo "  3. Log in to GitHub: gh auth login"
echo ""
echo "💡 CONVENIENCE TOOLS INSTALLED:"
echo "  - 'gh' for GitHub CLI operations (you can push/pull after gh auth login)"
echo "  - 'claude' for Claude Code CLI"
