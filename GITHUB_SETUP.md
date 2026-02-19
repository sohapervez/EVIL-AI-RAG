# GitHub Setup Guide

Follow these steps to push your RAG code to GitHub.

## Step 1: Create a GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the **"+"** icon in the top right → **"New repository"**
3. Fill in:
   - **Repository name**: `evil-ai-rag` (or your preferred name)
   - **Description**: "RAG chatbot for EVIL-AI research papers"
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have files)
4. Click **"Create repository"**

## Step 2: Initialize Git Repository (Local)

Open your terminal in the RAG project directory and run:

```bash
cd /Users/xrsope/Desktop/development/RAG

# Initialize git repository
git init

# Add all files (respects .gitignore)
git add .

# Make your first commit
git commit -m "Initial commit: EVIL-AI RAG system"
```

## Step 3: Connect to GitHub

After creating the repository, GitHub will show you commands. Use these:

```bash
# Add the remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/evil-ai-rag.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

**Note**: If you're using SSH instead of HTTPS:
```bash
git remote add origin git@github.com:YOUR_USERNAME/evil-ai-rag.git
```

## Step 4: Authentication

If prompted for credentials:
- **Username**: Your GitHub username
- **Password**: Use a **Personal Access Token** (not your GitHub password)
  - Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
  - Generate new token with `repo` scope
  - Copy and use it as password

## What Gets Pushed?

The `.gitignore` file ensures these are **NOT** pushed:
- ✅ `.env.local` (your API keys - **kept private**)
- ✅ `vectorstore/` (your indexed data)
- ✅ `.venv/` (virtual environment)
- ✅ `__pycache__/` (Python cache)

These **WILL** be pushed:
- ✅ All Python code (`app1.py`, `core/`, `ingest.py`, etc.)
- ✅ `README.md`
- ✅ `requirements.txt`
- ✅ `.env.local.template` (safe template without keys)
- ✅ `.gitignore`

## Future Updates

To push changes later:

```bash
# Check what changed
git status

# Add changed files
git add .

# Commit with a message
git commit -m "Description of your changes"

# Push to GitHub
git push
```

## Troubleshooting

### "Permission denied" error?
- Make sure you're authenticated with GitHub
- Use Personal Access Token instead of password
- Check that you have write access to the repository

### "Repository not found" error?
- Verify the repository name matches exactly
- Check that the repository exists on GitHub
- Ensure you're using the correct username

### Want to exclude PDFs?
If you don't want to commit the PDF files in `data/papers/`, uncomment this line in `.gitignore`:
```
# data/papers/*.pdf
```
Remove the `#` to activate it.

## Security Reminder

⚠️ **IMPORTANT**: Never commit:
- API keys or secrets
- `.env.local` file (already in .gitignore)
- Personal data or sensitive information

The `.gitignore` file protects your `.env.local` file, but always double-check before committing!
