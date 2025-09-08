#!/bin/bash
# Docker Credential Fix Script
# Docker 認證修復腳本

echo "🔧 Docker Credential Fix"
echo "======================="

echo "📍 Current Docker config location:"
echo "Windows: $USERPROFILE/.docker/"
echo "WSL: $HOME/.docker/"

# Check Docker config
if [ -f "$HOME/.docker/config.json" ]; then
    echo "✅ Found Docker config at $HOME/.docker/config.json"
    echo "📄 Current config:"
    cat "$HOME/.docker/config.json" | head -10
else
    echo "⚠️  No Docker config found, creating basic config..."
    mkdir -p "$HOME/.docker"
    cat > "$HOME/.docker/config.json" << 'EOF'
{
    "auths": {},
    "credsStore": ""
}
EOF
    echo "✅ Created basic Docker config"
fi

echo ""
echo "🔧 Fixing credential store issue..."

# Fix credential store in config
if [ -f "$HOME/.docker/config.json" ]; then
    # Remove or disable credential store
    python3 << 'PYTHON_SCRIPT'
import json
import os

config_path = os.path.expanduser("~/.docker/config.json")
try:
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Remove problematic credential store
    if 'credsStore' in config:
        print(f"Removing credsStore: {config['credsStore']}")
        del config['credsStore']
    
    if 'credHelpers' in config:
        print(f"Removing credHelpers: {config['credHelpers']}")
        del config['credHelpers']
    
    # Ensure auths exists
    if 'auths' not in config:
        config['auths'] = {}
    
    # Write back
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Fixed Docker credential config")
    
except Exception as e:
    print(f"❌ Error fixing config: {e}")
PYTHON_SCRIPT
fi

echo ""
echo "🧪 Testing Docker login (should not require credentials for public images)..."
if docker info > /dev/null 2>&1; then
    echo "✅ Docker connection working"
else
    echo "❌ Docker connection still failing"
    echo "嘗試重新啟動 Docker Desktop"
fi

echo ""
echo "🐳 Testing image pull without credentials..."
if docker pull hello-world:latest > /dev/null 2>&1; then
    echo "✅ Can pull public images successfully"
    docker rmi hello-world:latest > /dev/null 2>&1
else
    echo "⚠️  Still having credential issues"
    echo ""
    echo "手動修復步驟："
    echo "1. 關閉 Docker Desktop"
    echo "2. 刪除 $HOME/.docker/config.json"
    echo "3. 重新啟動 Docker Desktop"
    echo "4. 再次嘗試部署"
fi

echo ""
echo "🔄 Docker credential fix completed"
echo "現在可以嘗試重新部署 AIFX"