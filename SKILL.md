# 🧠 IBM AI Engineering — Master Skill Reference
> Personal workflow guide for converting, organising, and syncing course notebooks

---

## 📁 Where Everything Lives

```
IBM_AI_Engineering/                         ← repo root
├── SKILL.md                                ← 📚 this file
├── USER_GUIDE.md                           ← 📖 step-by-step guide
├── .vscode/
│   └── tasks.json                          ← ⌨️  VS Code shortcuts
├── scripts/
│   ├── CLAUDE_PROMPT.md                    ← 🤖 paste in Claude each chat
│   └── sync_notebook.py                    ← 🐍 auto-sync to GitHub
├── 01_ML_with_Python/
├── 02_Deep_Learning_Keras/                 ← 📍 you are here
├── 03_Scalable_ML_Apache_Spark/
├── 04_Deep_Learning_PyTorch/
├── 05_AI_Capstone_Deep_Learning/
├── 06_ML_Capstone/
├── 07_GenAI_LLMs_Architecture/
├── 08_GenAI_Foundational_Models_NLP/
├── 09_GenAI_Language_Modeling_Transformers/
├── 10_GenAI_Engineering_Fine_Tuning/
├── 11_GenAI_Advanced_Fine_Tuning_LLMs/
├── 12_AI_Agents_RAG_LangChain/
└── 13_Project_GenAI_Apps_RAG_LangChain/
```

---

## ⚡ Quick Reference — Per Lab Workflow

| Step | Action | Time |
|------|--------|------|
| 1 | Open Claude → paste `CLAUDE_PROMPT.md` | 10 sec |
| 2 | Upload raw `.ipynb` lab file | 5 sec |
| 3 | Download converted notebook | 5 sec |
| 4 | Open in VS Code | 5 sec |
| 5 | Press `Cmd+Shift+B` to sync to GitHub | 10 sec |
| **Total** | | **~35 sec** |

---

## 🤖 Claude Prompt Rules (summary)

Located at: `scripts/CLAUDE_PROMPT.md`

Key rules Claude follows:
- ✅ First person ("I build", "I implement")
- ✅ Removes lab/duration/IBM branding
- ✅ Adds Data & AI emojis
- ✅ LaTeX math formulas
- ✅ Full narrative markdown between code cells
- ✅ Fills in all solutions
- ✅ Summary table + Sandbox section at end

---

## 🐍 Sync Script Reference

Located at: `scripts/sync_notebook.py`

```bash
# Basic usage — auto-detects chapter
python scripts/sync_notebook.py ~/Downloads/MyNotebook.ipynb

# With custom commit message
python scripts/sync_notebook.py ~/Downloads/MyNotebook.ipynb "🧠 Custom message"
```

What it does automatically:
1. ✅ Validates the `.ipynb` file
2. ✅ Detects the right chapter folder from content
3. ✅ Copies notebook to the correct chapter folder
4. ✅ Creates `.py` mirror via Jupytext
5. ✅ Commits and pushes to GitHub

---

## ⌨️ VS Code Shortcuts

Located at: `.vscode/tasks.json`

| Shortcut | Action |
|----------|--------|
| `Cmd+Shift+B` | 🚀 Sync current notebook to GitHub |
| `Cmd+Shift+P` → "Run Task" → select | Access all tasks |

Available tasks:
- 🚀 Sync Notebook to GitHub
- 📥 Install Dependencies
- 📊 Git Status
- 📜 Git Log (last 10 commits)

---

## 🔧 Git Quick Reference

```bash
# Navigate to repo
cd ~/claude-workspace/learning/IBM_AI_Engineering

# Check status
git status

# Manual sync (if not using script)
git add .
git commit -m "🧠 Your message"
git push origin main

# See last 10 commits
git log --oneline -10

# Pull latest from GitHub
git pull origin main
```

---

## 📦 Dependencies

```bash
# Install once
pip install jupytext nbconvert
```

| Package | Purpose |
|---------|---------|
| `jupytext` | Pairs `.ipynb` ↔ `.py` for clean Git diffs |
| `nbconvert` | Export notebooks to PDF/HTML |

---

## 🌐 Important Links

| Resource | URL |
|----------|-----|
| GitHub Repo | https://github.com/IsReda/IBM-AI-Engineering |
| Coursera Course | https://www.coursera.org/professional-certificates/ai-engineer |
| Claude.ai | https://claude.ai |

---

## 📍 Current Progress

- [x] 01 · ML with Python
- [ ] 02 · Deep Learning & Keras  ← **you are here**
- [ ] 03 · Scalable ML on Apache Spark
- [ ] 04 · Deep Learning with PyTorch
- [ ] 05 · AI Capstone with Deep Learning
- [ ] 06 · ML Capstone
- [ ] 07 · GenAI & LLMs Architecture
- [ ] 08 · GenAI Foundational Models & NLP
- [ ] 09 · GenAI Language Modeling with Transformers
- [ ] 10 · GenAI Engineering & Fine-Tuning
- [ ] 11 · GenAI Advanced Fine-Tuning for LLMs
- [ ] 12 · AI Agents with RAG & LangChain
- [ ] 13 · Project: GenAI Apps with RAG & LangChain
