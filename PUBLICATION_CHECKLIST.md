# Publication Checklist for NTMM

Use this checklist before publishing your NTMM project or models.

## Code Quality
- [x] All tests pass (`pytest tests/ -v`)
- [x] Code is linted (`ruff check src tests`)
- [x] Code is formatted (`ruff format src tests`)
- [x] No sensitive data in repository
- [x] API keys and secrets in .gitignore
- [x] Type hints where appropriate
- [x] Docstrings for public functions

## Documentation
- [x] README.md is comprehensive and up-to-date
- [x] CHANGELOG.md reflects current version
- [x] LICENSE file present with correct copyright
- [x] CONTRIBUTING.md explains how to contribute
- [x] CODE_OF_CONDUCT.md present
- [x] SECURITY.md with security policy
- [x] FAQ.md answers common questions
- [x] DEPLOYMENT.md covers production deployment
- [x] Examples are working and documented
- [x] CITATION.cff for academic citations

## Branding & Ownership
- [x] Project renamed to NTMM
- [x] All references updated to NorthernTribe Medical Models
- [x] Copyright assigned to NorthernTribe Research
- [x] Model cards include NorthernTribe attribution
- [x] Student models saved to `ntmm-student` directory
- [x] Model prefix set to "NTMM" in config

## Configuration
- [x] mcp.json has sensible defaults
- [x] Student model path uses NTMM branding
- [x] Dataset configurations are valid
- [x] Hyperparameters are documented
- [ ] Update repository URL in all docs (replace `<this-repo>`)
- [ ] Update contact email in SECURITY.md

## Testing
- [x] Unit tests for core functionality
- [x] Integration tests for pipeline
- [x] Tests run in CI/CD
- [x] Test coverage is reasonable
- [ ] Run full pipeline test: `./run_all_steps.sh quick`
- [ ] Verify model card generation
- [ ] Test inference example

## CI/CD
- [x] GitHub Actions workflows present
- [x] Test workflow runs on push/PR
- [x] Publish workflow ready (needs PyPI token)
- [x] Python version matrix includes 3.10, 3.11, 3.12
- [x] Caching configured for faster builds

## Repository Setup
- [ ] Create GitHub repository
- [ ] Add repository description
- [ ] Add topics/tags: `medical-nlp`, `knowledge-distillation`, `pytorch`, `transformers`
- [ ] Enable Issues
- [ ] Enable Discussions (optional)
- [ ] Add repository URL to pyproject.toml
- [ ] Add repository URL to CITATION.cff

## GitHub Configuration
- [ ] Set up branch protection for main
- [ ] Require PR reviews
- [ ] Require status checks to pass
- [ ] Add CODEOWNERS file (optional)
- [ ] Configure GitHub Pages for docs (optional)

## Secrets & Tokens
- [ ] Add PYPI_API_TOKEN to GitHub secrets (for publishing)
- [ ] Remove any hardcoded credentials
- [ ] Verify kaggle.json is in .gitignore
- [ ] Check for any API keys in code

## Model Publishing (Hugging Face)
- [ ] Create Hugging Face organization: NorthernTribe-Research
- [ ] Train a model: `./run_all_steps.sh`
- [ ] Verify model card: `cat saved_models/ntmm-student/README.md`
- [ ] Test model locally: `python examples/inference_example.py`
- [ ] Upload to Hub: `huggingface-cli upload NorthernTribe-Research/ntmm-v1 saved_models/ntmm-student/`
- [ ] Add model card tags on Hub
- [ ] Test model from Hub

## Package Publishing (PyPI)
- [ ] Create PyPI account
- [ ] Generate API token
- [ ] Test build: `python -m build`
- [ ] Test package: `twine check dist/*`
- [ ] Upload to TestPyPI first
- [ ] Test install from TestPyPI
- [ ] Upload to PyPI: `twine upload dist/*`
- [ ] Test install: `pip install ntmm`

## Communication
- [ ] Write release notes
- [ ] Update CHANGELOG.md
- [ ] Create GitHub release
- [ ] Announce on relevant platforms (optional)
- [ ] Share on social media (optional)

## Legal & Compliance
- [x] License is appropriate (MIT)
- [x] Copyright notices are correct
- [ ] Verify dataset licenses allow your use case
- [ ] Ensure no patient data or PHI in repository
- [ ] Review terms of service for platforms used
- [ ] Consider trademark for "NTMM" (optional)

## Final Checks
- [ ] All URLs in documentation are correct
- [ ] All links work (no 404s)
- [ ] Images/screenshots are included if needed
- [ ] Version numbers are consistent
- [ ] No TODO comments in production code
- [ ] No debug print statements
- [ ] No commented-out code blocks

## Post-Publication
- [ ] Monitor GitHub issues
- [ ] Respond to community feedback
- [ ] Update documentation based on questions
- [ ] Plan next version features
- [ ] Set up monitoring for deployed models

---

## Quick Verification Commands

```bash
# Test everything
pytest tests/ -v
ruff check src tests
ruff format --check src tests

# Build package
python -m build
twine check dist/*

# Test pipeline
./run_all_steps.sh quick

# Test inference
python examples/inference_example.py

# Check for secrets
git secrets --scan  # if installed
grep -r "api_key\|password\|secret" src/ tests/
```

## Ready to Publish?

Once all items are checked:
1. Create a git tag: `git tag -a v0.1.0 -m "Initial release"`
2. Push tag: `git push origin v0.1.0`
3. Create GitHub release from tag
4. Publish to PyPI (if desired)
5. Announce your NTMM models!

---

**Remember**: You own the NTMM brand and models. Make sure all attribution is correct before publishing!
