# Upload this package to GitHub

1. Open https://github.com/shreyas250825/Explanable-AI-Research-Work.
2. To upload in the browser, choose **Add file → Upload files** and upload the contents of this ZIP **preserving folder paths**. For multi-folder upload, a Git command is generally more reliable.
3. If cloning an existing repository, **merge these files** instead of deleting work already there. Review the README before replacing one.
4. Check that `README.md` renders its Mermaid diagrams, `src/` and `docs/` are intact, and GitHub Actions tests pass.
5. Do **not** upload local IQ raw files, dataset CSVs, model weights, API tokens, private metadata, or full original notebooks accidentally.
6. Add the dataset link to `data/README.md` and the paper availability statement **only after** you have actually published a permitted dataset archive.

Suggested Git workflow (run from the existing cloned repository, after copying the package contents into it):

```bash
git add README.md docs src scripts tests data/README.md requirements.txt requirements-ml.txt .gitignore .github/workflows/tests.yml UPLOAD_TO_GITHUB.md
git commit -m "Add documented partial wireless XAI research code"
git push
```

No `LICENSE` was included because the authors have not selected one. Choose a license deliberately before granting reuse rights.
