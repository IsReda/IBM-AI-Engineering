# 📖 IBM AI Engineering — Complete User Guide
> Everything you need to know, step by step, from zero to GitHub

---

## 🗺️ Table of Contents

1. [One-Time Setup](#1-one-time-setup)
2. [Per-Lab Workflow](#2-per-lab-workflow)
3. [Using Claude](#3-using-claude)
4. [Using the Sync Script](#4-using-the-sync-script)
5. [Using VS Code Tasks](#5-using-vs-code-tasks)
6. [Troubleshooting](#6-troubleshooting)
7. [Quick Command Reference](#7-quick-command-reference)

---

## 1 · One-Time Setup

> Do this once. Never again.

### Step 1.1 · Install dependencies
Open Terminal and run:
```bash
pip install jupytext nbconvert
```

### Step 1.2 · Copy the pipeline files into your repo
```bash
# Copy scripts
cp /path/to/downloaded/CLAUDE_PROMPT.md \
   ~/claude-workspace/learning/IBM_AI_Engineering/scripts/

cp /path/to/downloaded/sync_notebook.py \
   ~/claude-workspace/learning/IBM_AI_Engineering/scripts/

# Copy VS Code task
cp /path/to/downloaded/tasks.json \
   ~/claude-workspace/learning/IBM_AI_Engineering/.vscode/

# Copy reference files
cp /path/to/downloaded/SKILL.md \
   ~/claude-workspace/learning/IBM_AI_Engineering/

cp /path/to/downloaded/USER_GUIDE.md \
   ~/claude-workspace/learning/IBM_AI_Engineering/
```

### Step 1.3 · Open your repo in VS Code
```bash
code ~/claude-workspace/learning/IBM_AI_Engineering
```
> 💡 Pin this to your VS Code — you will open it every study session

### Step 1.4 · Push the pipeline files to GitHub
```bash
cd ~/claude-workspace/learning/IBM_AI_Engineering
git add .
git commit -m "⚙️ Add IBM AI pipeline scripts and guides"
git push origin main
```

---

## 2 · Per-Lab Workflow

> Do this for every new IBM lab. Takes ~35 seconds.

```
┌─────────────────────────────────────────────────┐
│  1. Open claude.ai in browser                   │
│  2. Paste CLAUDE_PROMPT.md as first message     │
│  3. Upload raw .ipynb lab file                  │
│  4. Download converted notebook                 │
│  5. Open notebook in VS Code                    │
│  6. Press Cmd+Shift+B → synced to GitHub ✅     │
└─────────────────────────────────────────────────┘
```

---

## 3 · Using Claude

### Step 3.1 · Open CLAUDE_PROMPT.md in VS Code
```
IBM_AI_Engineering/
└── scripts/
    └── CLAUDE_PROMPT.md    ← open this
```

### Step 3.2 · Copy all content below the divider line (`---`)

### Step 3.3 · Go to claude.ai → start a new chat

### Step 3.4 · Paste the prompt as your FIRST message and send it

### Step 3.5 · Upload your raw IBM lab `.ipynb` file
> Drag and drop it into the Claude chat window

### Step 3.6 · Wait ~30 seconds

### Step 3.7 · Click the download link Claude provides
> Save the file to your Downloads folder

---

## 4 · Using the Sync Script

The script auto-detects which chapter folder to use based on the notebook content.

### Basic usage
```bash
cd ~/claude-workspace/learning/IBM_AI_Engineering
python scripts/sync_notebook.py ~/Downloads/YourNotebook.ipynb
```

### With a custom commit message
```bash
python scripts/sync_notebook.py ~/Downloads/YourNotebook.ipynb "🧠 My custom message"
```

### What you will see
```
🚀 IBM AI Engineering — Smart Sync
────────────────────────────────────
📎 Notebook: ANN_Forward_Propagation.ipynb

📋 Step 1 · Validating notebook...
  ✅ Valid notebook — 24 cells

🔍 Step 2 · Detecting chapter...
  🎯 Auto-detected chapter: 02_Deep_Learning_Keras (score: 4)

📁 Step 3 · Copying to 02_Deep_Learning_Keras/...
  ✅ Saved → .../02_Deep_Learning_Keras/ANN_Forward_Propagation.ipynb

🔁 Step 4 · Jupytext pairing...
  ✅ .py mirror created

📤 Step 5 · Pushing to GitHub...
  ✅ Done! Pushed to GitHub
```

### If chapter detection fails
The script will show you a numbered menu — just type the number:
```
⚠️  Could not auto-detect chapter. Please choose:
   1. 01_ML_with_Python
   2. 02_Deep_Learning_Keras
   ...
Enter number: 2
```

---

## 5 · Using VS Code Tasks

> The fastest way — no Terminal needed

### Step 5.1 · Open a notebook in VS Code
Click any `.ipynb` file in the Explorer panel

### Step 5.2 · Press `Cmd+Shift+B`
A Terminal panel opens at the bottom and the sync runs automatically

### Step 5.3 · Watch the output
You will see the same output as the sync script above

### Other available tasks
Press `Cmd+Shift+P` → type "Run Task" → choose from:
- 🚀 Sync Notebook to GitHub
- 📥 Install Dependencies
- 📊 Git Status
- 📜 Git Log

---

## 6 · Troubleshooting

### ❌ "fatal: not a git repository"
```bash
cd ~/claude-workspace/learning/IBM_AI_Engineering
git init
git remote add origin https://github.com/IsReda/IBM-AI-Engineering.git
git pull origin main
```

### ❌ "remote origin already exists"
```bash
git remote remove origin
git remote add origin https://github.com/IsReda/IBM-AI-Engineering.git
```

### ❌ "jupytext: command not found"
```bash
pip install jupytext
```

### ❌ Chapter detected wrongly
Run with a custom message and move the file manually:
```bash
mv 02_Deep_Learning_Keras/WrongFile.ipynb 03_Scalable_ML_Apache_Spark/
git add .
git commit -m "📁 Move notebook to correct chapter"
git push origin main
```

### ❌ Push rejected
```bash
git pull origin main --rebase
git push origin main
```

### ❌ Claude generated wrong format
Make sure you pasted the FULL content of `CLAUDE_PROMPT.md` — everything below the `---` divider line — as your FIRST message before uploading the file.

---

## 7 · Quick Command Reference

```bash
# Navigate to repo
cd ~/claude-workspace/learning/IBM_AI_Engineering

# Sync a notebook
python scripts/sync_notebook.py ~/Downloads/Notebook.ipynb

# Manual git push
git add . && git commit -m "🧠 message" && git push origin main

# Check status
git status

# See history
git log --oneline -10

# Open in VS Code
code .

# Install dependencies
pip install jupytext nbconvert
```

---

## 💡 Pro Tips

1. **Keep Claude chat open** during a study session — the prompt is already set, just upload each new lab
2. **Run cells top to bottom** in VS Code after downloading — make sure everything executes cleanly before pushing
3. **Update the progress checklist** in `SKILL.md` as you complete each chapter
4. **Use meaningful commit messages** — your GitHub becomes your learning portfolio

---

*Last updated: May 2026 · IsReda · IBM AI Engineering Professional Certificate*
