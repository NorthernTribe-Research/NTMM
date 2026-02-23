# Push NTMM to GitHub

## Step-by-Step Guide

### 1. Create GitHub Repository

Go to GitHub and create a new repository:
- Organization: **NorthernTribe-Research**
- Repository name: **NTMM**
- Description: "NorthernTribe Medical Models - Knowledge distillation pipeline for medical reasoning"
- Visibility: Public (or Private if you prefer)
- **DO NOT** initialize with README, .gitignore, or license (we already have these)

### 2. Initialize Git (if not already done)

```bash
# Check if git is initialized
git status

# If not initialized, run:
git init
```

### 3. Stage All Files

```bash
# Add all files to git
git add .

# Check what will be committed
git status
```

### 4. Create Initial Commit

```bash
# Commit with a meaningful message
git commit -m "Initial commit: NTMM v0.1.0 - NorthernTribe Medical Models

- Complete knowledge distillation pipeline
- Branded NTMM student models
- Comprehensive documentation (900+ lines)
- Production deployment guides
- CI/CD workflows
- Examples and tests
"
```

### 5. Add Remote and Push

```bash
# Add GitHub remote
git remote add origin https://github.com/NorthernTribe-Research/NTMM.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

### 6. Verify on GitHub

Visit: https://github.com/NorthernTribe-Research/NTMM

You should see:
- ✅ All files uploaded
- ✅ README.md displayed on homepage
- ✅ Badges showing in README
- ✅ License file recognized

### 7. Configure Repository Settings

On GitHub, go to Settings:

#### General
- ✅ Description: "NorthernTribe Medical Models - Knowledge distillation pipeline for medical reasoning"
- ✅ Website: (optional)
- ✅ Topics: Add tags
  - `medical-nlp`
  - `knowledge-distillation`
  - `pytorch`
  - `transformers`
  - `ntmm`
  - `medical-ai`

#### Features
- ✅ Enable Issues
- ✅ Enable Discussions (optional)
- ✅ Disable Wiki (we have docs in repo)
- ✅ Disable Projects (unless you want to use them)

#### Branches
- ✅ Set `main` as default branch
- ✅ Add branch protection rules (optional but recommended):
  - Require pull request reviews
  - Require status checks to pass
  - Require branches to be up to date

#### Secrets and Variables
Add these for CI/CD:
- `PYPI_API_TOKEN` (if you want to publish to PyPI)

### 8. Create First Release

1. Go to Releases → "Create a new release"
2. Click "Choose a tag" → Type `v0.1.0` → "Create new tag"
3. Release title: `NTMM v0.1.0 - Initial Release`
4. Description: Copy from CHANGELOG.md
5. Click "Publish release"

### 9. Test CI/CD

The GitHub Actions should automatically run:
- Test workflow (on push)
- Check the Actions tab to see results

### 10. Update Local Repository

```bash
# Pull any changes (like release tags)
git pull origin main

# Check everything is synced
git status
```

## Quick Command Summary

```bash
# One-liner to push (if git is already initialized)
git add . && \
git commit -m "Initial commit: NTMM v0.1.0" && \
git remote add origin https://github.com/NorthernTribe-Research/NTMM.git && \
git branch -M main && \
git push -u origin main
```

## Troubleshooting

### "remote origin already exists"
```bash
# Remove existing remote
git remote remove origin

# Add the correct one
git remote add origin https://github.com/NorthernTribe-Research/NTMM.git
```

### "failed to push some refs"
```bash
# If remote has changes, pull first
git pull origin main --rebase

# Then push
git push -u origin main
```

### Authentication Issues
```bash
# Use GitHub CLI (recommended)
gh auth login

# Or use personal access token
# Generate at: https://github.com/settings/tokens
```

## After Pushing

### Clone Test
Test that others can clone:
```bash
cd /tmp
git clone https://github.com/NorthernTribe-Research/NTMM.git
cd NTMM
./setup.sh
```

### Update README Badge (optional)
Add build status badge to README.md:
```markdown
[![Tests](https://github.com/NorthernTribe-Research/NTMM/workflows/Test/badge.svg)](https://github.com/NorthernTribe-Research/NTMM/actions)
```

## Next Steps After GitHub

1. **Publish Model to Hugging Face Hub**
   ```bash
   ./run_all_steps.sh quick
   pip install huggingface_hub
   huggingface-cli login
   huggingface-cli upload NorthernTribe-Research/ntmm-v1 saved_models/ntmm-student/
   ```

2. **Announce Your Project**
   - Share on social media
   - Post on relevant forums
   - Submit to awesome lists

3. **Monitor and Maintain**
   - Watch for issues
   - Respond to pull requests
   - Update documentation as needed

## Verification Checklist

After pushing, verify:
- [ ] Repository is public/accessible
- [ ] README displays correctly
- [ ] All files are present
- [ ] License is recognized by GitHub
- [ ] Topics/tags are added
- [ ] CI/CD workflows run successfully
- [ ] Issues are enabled
- [ ] First release is created (v0.1.0)

---

**Repository URL**: https://github.com/NorthernTribe-Research/NTMM

**Congratulations!** Your NTMM project is now live on GitHub! 🎉
