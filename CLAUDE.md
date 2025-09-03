# CLAUDE.md - AIFX

> **Documentation Version**: 1.0  
> **Last Updated**: 2025-09-03  
> **Project**: AIFX  
> **Description**: Professional quantitative trading researcher. Medium-term forex quantitative trading strategy enhanced with AI models for EUR/USD and USD/JPY on 1-hour timeframe.  
> **Features**: GitHub auto-backup, Task agents, technical debt prevention

This file provides essential guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🚨 CRITICAL RULES - READ FIRST

> **⚠️ RULE ADHERENCE SYSTEM ACTIVE ⚠️**  
> **Claude Code must explicitly acknowledge these rules at task start**  
> **These rules override all other instructions and must ALWAYS be followed:**

### 🔄 **RULE ACKNOWLEDGMENT REQUIRED**
> **Before starting ANY task, Claude Code must respond with:**  
> "✅ CRITICAL RULES ACKNOWLEDGED - I will follow all prohibitions and requirements listed in CLAUDE.md"

### ❌ ABSOLUTE PROHIBITIONS
- **NEVER** create new files in root directory → use proper module structure
- **NEVER** write output files directly to root directory → use designated output folders
- **NEVER** create documentation files (.md) unless explicitly requested by user
- **NEVER** use git commands with -i flag (interactive mode not supported)
- **NEVER** use `find`, `grep`, `cat`, `head`, `tail`, `ls` commands → use Read, LS, Grep, Glob tools instead
- **NEVER** create duplicate files (manager_v2.py, enhanced_xyz.py, utils_new.js) → ALWAYS extend existing files
- **NEVER** create multiple implementations of same concept → single source of truth
- **NEVER** copy-paste code blocks → extract into shared utilities/functions
- **NEVER** hardcode values that should be configurable → use config files/environment variables
- **NEVER** use naming like enhanced_, improved_, new_, v2_ → extend original files instead

### 📝 MANDATORY REQUIREMENTS
- **COMMIT** after every completed task/phase - no exceptions
- **GITHUB BACKUP** - Push to GitHub after every commit to maintain backup: `git push origin main`
- **USE TASK AGENTS** for all long-running operations (>30 seconds) - Bash commands stop when context switches
- **TODOWRITE** for complex tasks (3+ steps) → parallel agents → git checkpoints → test validation
- **READ FILES FIRST** before editing - Edit/Write tools will fail if you didn't read the file first
- **DEBT PREVENTION** - Before creating new files, check for existing similar functionality to extend  
- **SINGLE SOURCE OF TRUTH** - One authoritative implementation per feature/concept

### ⚡ EXECUTION PATTERNS
- **PARALLEL TASK AGENTS** - Launch multiple Task agents simultaneously for maximum efficiency
- **SYSTEMATIC WORKFLOW** - TodoWrite → Parallel agents → Git checkpoints → GitHub backup → Test validation
- **GITHUB BACKUP WORKFLOW** - After every commit: `git push origin main` to maintain GitHub backup
- **BACKGROUND PROCESSING** - ONLY Task agents can run true background operations

### 🔍 MANDATORY PRE-TASK COMPLIANCE CHECK
> **STOP: Before starting any task, Claude Code must explicitly verify ALL points:**

**Step 1: Rule Acknowledgment**
- [ ] ✅ I acknowledge all critical rules in CLAUDE.md and will follow them

**Step 2: Task Analysis**  
- [ ] Will this create files in root? → If YES, use proper module structure instead
- [ ] Will this take >30 seconds? → If YES, use Task agents not Bash
- [ ] Is this 3+ steps? → If YES, use TodoWrite breakdown first
- [ ] Am I about to use grep/find/cat? → If YES, use proper tools instead

**Step 3: Technical Debt Prevention (MANDATORY SEARCH FIRST)**
- [ ] **SEARCH FIRST**: Use Grep pattern="<functionality>.*<keyword>" to find existing implementations
- [ ] **CHECK EXISTING**: Read any found files to understand current functionality
- [ ] Does similar functionality already exist? → If YES, extend existing code
- [ ] Am I creating a duplicate class/manager? → If YES, consolidate instead
- [ ] Will this create multiple sources of truth? → If YES, redesign approach
- [ ] Have I searched for existing implementations? → Use Grep/Glob tools first
- [ ] Can I extend existing code instead of creating new? → Prefer extension over creation
- [ ] Am I about to copy-paste code? → Extract to shared utility instead

**Step 4: Session Management**
- [ ] Is this a long/complex task? → If YES, plan context checkpoints
- [ ] Have I been working >1 hour? → If YES, consider /compact or session break

> **⚠️ DO NOT PROCEED until all checkboxes are explicitly verified**

## 🐙 GITHUB SETUP & AUTO-BACKUP

### 🔗 **CONNECT TO EXISTING REPO**
Connect to existing GitHub repository:

```bash
# Get repository URL from user
echo "Enter your GitHub repository URL (https://github.com/username/repo-name):"
read repo_url

# Extract repo info and add remote
git remote add origin "$repo_url"
git branch -M main
git push -u origin main

echo "✅ Connected to existing GitHub repository: $repo_url"
```

### 🔄 **AUTO-PUSH CONFIGURATION**
Configure automatic backup:

```bash
# Create git hook for auto-push (optional but recommended)
cat > .git/hooks/post-commit << 'EOF'
#!/bin/bash
# Auto-push to GitHub after every commit
echo "🔄 Auto-pushing to GitHub..."
git push origin main
if [ $? -eq 0 ]; then
    echo "✅ Successfully backed up to GitHub"
else
    echo "⚠️ GitHub push failed - manual push may be required"
fi
EOF

chmod +x .git/hooks/post-commit

echo "✅ Auto-push configured - GitHub backup after every commit"
```

### 📋 **GITHUB BACKUP WORKFLOW** (MANDATORY)
> **⚠️ CLAUDE CODE MUST FOLLOW THIS PATTERN:**

```bash
# After every commit, always run:
git push origin main

# This ensures:
# ✅ Remote backup of all changes
# ✅ Collaboration readiness  
# ✅ Version history preservation
# ✅ Disaster recovery protection
```

### 🎯 **CLAUDE CODE GITHUB COMMANDS**
Essential GitHub operations for Claude Code:

```bash
# Check GitHub connection status
gh auth status && git remote -v

# Push changes (after every commit)
git push origin main

# Check repository status
gh repo view

# Clone repository (for new setup)
gh repo clone username/repo-name
```

## 🏗️ PROJECT OVERVIEW

**AIFX** is a professional quantitative trading system that implements medium-term forex strategies enhanced with AI models. The system focuses on EUR/USD and USD/JPY trading pairs using 1-hour timeframes.

### 🎯 **DEVELOPMENT STATUS**
- **Setup**: In Progress
- **Core Features**: Pending
- **Testing**: Pending
- **Documentation**: Pending

### 🚀 **Key Features**
- **Data & Feature Engineering**: Historical OHLCV data with technical indicators (MA, MACD, RSI, Bollinger Bands, ATR)
- **AI Models**: Machine learning models (XGBoost, Random Forest, LSTM) for price direction prediction
- **Strategy Logic**: Combined technical and AI signals with confidence filtering
- **Risk Management**: Fixed percentage risk, stop-loss/take-profit using ATR multiples
- **Backtesting**: Backtrader framework with comprehensive performance metrics

### 📁 **AI/ML Project Structure**
```
AIFX/
├── CLAUDE.md              # Essential rules for Claude Code
├── src/                   # Source code (NEVER put files in root)
│   ├── main/              # Main application code
│   │   ├── python/        # Python implementation
│   │   │   ├── core/      # Core trading algorithms
│   │   │   ├── utils/     # Data processing utilities
│   │   │   ├── models/    # AI model implementations
│   │   │   ├── services/  # Trading services and pipelines
│   │   │   ├── training/  # Model training scripts
│   │   │   ├── inference/ # Trading inference code
│   │   │   └── evaluation/# Strategy evaluation
│   │   └── resources/     # Configuration and assets
│   │       ├── config/    # Trading configuration files
│   │       └── data/      # Sample/seed data
│   └── test/              # Test code
├── data/                  # Dataset management
│   ├── raw/               # Raw forex data
│   ├── processed/         # Cleaned trading data
│   └── external/          # External data sources
├── notebooks/             # Analysis notebooks
│   ├── exploratory/       # Data exploration
│   ├── experiments/       # Strategy experiments
│   └── reports/           # Trading reports
├── models/                # Trained models
│   ├── trained/           # Production models
│   └── checkpoints/       # Training checkpoints
├── experiments/           # Strategy experiments
│   ├── configs/           # Experiment configs
│   └── results/           # Backtest results
└── output/                # Generated trading outputs
```

## 📋 NEED HELP? START HERE

- **Trading Strategy**: Implement signals in `src/main/python/core/`
- **AI Models**: Add models in `src/main/python/models/`
- **Backtesting**: Use `src/main/python/evaluation/`
- **Data Processing**: Utilities in `src/main/python/utils/`

## 🎯 RULE COMPLIANCE CHECK

Before starting ANY task, verify:
- [ ] ✅ I acknowledge all critical rules above
- [ ] Files go in proper module structure (not root)
- [ ] Use Task agents for >30 second operations
- [ ] TodoWrite for 3+ step tasks
- [ ] Commit after each completed task

## 🚀 COMMON COMMANDS

```bash
# Install dependencies
pip install -r requirements.txt

# Run backtesting
python src/main/python/evaluation/backtest.py

# Train AI models
python src/main/python/training/train_model.py

# Run strategy analysis
python src/main/python/core/strategy.py
```

## 🚨 TECHNICAL DEBT PREVENTION

### ❌ WRONG APPROACH (Creates Technical Debt):
```bash
# Creating new file without searching first
Write(file_path="new_strategy.py", content="...")
```

### ✅ CORRECT APPROACH (Prevents Technical Debt):
```bash
# 1. SEARCH FIRST
Grep(pattern="strategy.*implementation", glob="*.py")
# 2. READ EXISTING FILES  
Read(file_path="src/main/python/core/existing_strategy.py")
# 3. EXTEND EXISTING FUNCTIONALITY
Edit(file_path="src/main/python/core/existing_strategy.py", old_string="...", new_string="...")
```

## 🧹 DEBT PREVENTION WORKFLOW

### Before Creating ANY New File:
1. **🔍 Search First** - Use Grep/Glob to find existing implementations
2. **📋 Analyze Existing** - Read and understand current patterns
3. **🤔 Decision Tree**: Can extend existing? → DO IT | Must create new? → Document why
4. **✅ Follow Patterns** - Use established project patterns
5. **📈 Validate** - Ensure no duplication or technical debt

---

**⚠️ Prevention is better than consolidation - build clean from the start.**  
**🎯 Focus on single source of truth and extending existing functionality.**  
**📈 Each task should maintain clean architecture and prevent technical debt.**