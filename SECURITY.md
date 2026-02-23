# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in NTMM, please report it by:

1. **Do NOT** open a public issue
2. Email security concerns to: [your-security-email@example.com]
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will respond within 48 hours and work with you to address the issue.

## Security Considerations

### Model Security
- NTMM models are trained on public medical datasets
- Models should be validated before clinical deployment
- Not intended as a replacement for professional medical advice

### Data Privacy
- No patient data or PHI should be used with this pipeline without proper authorization
- Ensure compliance with HIPAA, GDPR, or relevant regulations in your jurisdiction

### Dependencies
- Keep dependencies updated: `pip install --upgrade -r requirements.txt`
- Review security advisories for PyTorch and Transformers regularly
